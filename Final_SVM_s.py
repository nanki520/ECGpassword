# ======================================================================
#  ECG_Auth_System.py      rev-I  (2025-04-24)
#  FINAL_SVM_S  (feature-based multiclass SVM)
# ======================================================================

import os, re
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, welch, resample_poly
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# -------------------- globals (mirror MATLAB "global") --------------------
SVMModel = None
mima = None
Xtrain_mean = None
Xtrain_std = None


# =================  辅助函数   =================
def resample_to(x, fs0, fs1):
    if fs0 == fs1:
        return np.asarray(x, dtype=np.float64)
    from fractions import Fraction
    frac = Fraction(fs1, fs0).limit_denominator(2000)
    return resample_poly(np.asarray(x, dtype=np.float64), frac.numerator, frac.denominator).astype(np.float64)


# 降噪 Denoising
def denoiseECG(x, Fs):
    low, high = 0.5, 40.0
    b, a = butter(4, [low/(Fs/2), high/(Fs/2)], btype="band")
    y = filtfilt(b, a, x).astype(np.float64)
    # y = (y - mean(y)) ./ std(y);   % MATLAB commented out
    return y


# —— 提取基本特征 extract basic features—— %
def extractFeatures(ecg, Fs):
    ecg = np.asarray(ecg, dtype=np.float64)

    # —— 时域特征 time-domain features—— %
    loc, _ = find_peaks(
        ecg,
        height=(np.mean(ecg) + 0.5*np.std(ecg)),
        distance=max(1, int(round(0.25*Fs)))
    )

    RR = np.diff(loc) / Fs if loc.size >= 2 else np.array([])
    if RR.size < 3:
        RR = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    feat_meanRR = np.nanmean(RR)
    feat_SDNN   = np.nanstd(RR)
    feat_RMSSD  = np.sqrt(np.nanmean(np.diff(RR)**2)) if RR.size >= 2 else np.nan

    # —— 波形相似度 similarity between waveforms—— %
    beatLen = int(round(0.7 * Fs))
    if loc.size >= 2:
        half = beatLen // 2
        tmpl = ecg[max(0, loc[0]-half):min(ecg.size, loc[0]+half)]
        nCmp = min(8, loc.size - 1)
        ccs = []
        for k in range(nCmp):
            seg = ecg[max(0, loc[k+1]-half):min(ecg.size, loc[k+1]+half)]
            L = min(len(tmpl), len(seg))
            if L >= 10 and np.std(tmpl[:L]) > 0 and np.std(seg[:L]) > 0:
                ccs.append(np.corrcoef(tmpl[:L], seg[:L])[0, 1])
        feat_shape = float(np.nanmean(ccs)) if len(ccs) else np.nan
    else:
        feat_shape = np.nan

    # —— 频域HRV特征 frequency-domain HRV features—— %
    f, pxx = welch(ecg, fs=Fs, window="hamming", nperseg=512, noverlap=256, nfft=1024, scaling="density")

    def bandpower(bnd):
        lo, hi = bnd
        m = (f >= lo) & (f <= hi)
        if not np.any(m):
            return 0.0
        return float(np.trapz(pxx[m], f[m]))

    feat_LF = bandpower((0.04, 0.15))
    feat_HF = bandpower((0.15, 0.40))
    feat_LFHF = feat_LF / max(feat_HF, np.finfo(np.float64).eps)

    vec = np.round(
        np.array([feat_meanRR, feat_SDNN, feat_RMSSD, feat_shape, feat_LF, feat_HF, feat_LFHF], dtype=np.float64),
        3
    )

    return {
        "meanRR": feat_meanRR,
        "SDNN": feat_SDNN,
        "RMSSD": feat_RMSSD,
        "shape": feat_shape,
        "LF": feat_LF,
        "HF": feat_HF,
        "LFHF": feat_LFHF,
        "vec": vec
    }


# —— generate a hashed password —— %
def generatePassword(vec):
    s = "_".join([f"{float(x):.3f}" for x in np.asarray(vec).ravel().tolist()])
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest().lower()


# ---------- 文件读取 file reading ----------
def loadAmpECG_manual(folder, rec, lead):
    hea_path = os.path.join(folder, rec + ".hea")
    dat_path = os.path.join(folder, rec + ".dat")

    with open(hea_path, "r", errors="ignore") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("#")]

    hdr = re.split(r"\s+", lines[0])
    nSig = int(float(hdr[1]))
    Fs = float(hdr[2])  # Sampling rate 采样率

    rows = lines[1:]
    rows = [ln for ln in rows if "filtered" not in ln.lower()]

    lead = max(1, min(int(lead), len(rows)))  # 1-based
    tok = re.split(r"\s+", rows[lead-1])

    # MATLAB uses tok{3} and tok{6} (1-based) => tok[2], tok[5]
    gain = float(tok[2])
    baseline = float(tok[5])

    raw = np.fromfile(dat_path, dtype="<i2").astype(np.float64)
    L = (raw.size // nSig) * nSig
    raw = raw[:L]
    sig = raw.reshape(nSig, -1)[lead-1, :]

    if gain != 0:
        sig = (sig - baseline) / gain

    # 小波处理 wavelet processing
    wname = "coif4"
    level = 7
    coeffs = pywt.wavedec(sig, wname, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # A7=0
    sig = pywt.waverec(coeffs, wname)

    return sig.astype(np.float64), Fs


# =================== AMP 分组初始化 ===================
def initUsersFromAMP(ampDir, Fs_target, band, leadIdx, winSec, db):
    heaFiles = [f for f in os.listdir(ampDir) if f.endswith(".hea")]
    recNames = sorted(list({os.path.splitext(f)[0] for f in heaFiles}))
    groups = {}

    for rec in recNames:
        m = re.match(r"^(.*?_\d+)_\d+$", rec)
        if not m:
            continue
        key = m.group(1)
        groups.setdefault(key, []).append(rec)

    keys = list(groups.keys())

    Xtrain = []
    labels = []
    userID = 1

    global mima, Xtrain_mean, Xtrain_std, SVMModel
    mima = []

    for n in range(0, max(0, len(keys) - 1)):
        recSet = groups[keys[n]]
        user_feats = []  # 每个用户单独积

        for r in recSet[0:len(recSet)]:
            ecg, Fs0 = loadAmpECG_manual(ampDir, r, leadIdx)
            if Fs0 != Fs_target:
                ecg = resample_to(ecg, Fs0, Fs_target)
            ecg = denoiseECG(ecg, Fs_target)

            step = int(Fs_target * winSec)

            if len(ecg) > 100000:
                for start_idx in range(0, len(ecg) - step, step):
                    ecg_seg = ecg[start_idx:start_idx + step]
                    feat = extractFeatures(ecg_seg, Fs_target)
                    vec = feat["vec"]

                    # 检查并处理NaN Check and handle NaN values
                    if np.any(np.isnan(vec)):
                        continue

                    Xtrain.append(vec)
                    labels.append(userID)
                    user_feats.append(vec)
            else:
                ecg_seg = ecg[:min(len(ecg), step)]
                if ecg_seg.size < step:
                    ecg_seg = np.pad(ecg_seg, (0, step - ecg_seg.size))
                feat = extractFeatures(ecg_seg, Fs_target)
                vec = feat["vec"]

                if np.any(np.isnan(vec)):
                    continue

                Xtrain.append(vec)
                labels.append(userID)
                user_feats.append(vec)

        # 生成密码 Generate password
        if len(user_feats) > 0:
            mean_feat = np.mean(np.vstack(user_feats), axis=0)
            pwd = generatePassword(mean_feat)
            mima.append(pwd)
            print(f"[USER] Registered user {userID}  ({len(user_feats)} segments)")
            userID += 1
        else:
            print(f"[WARN] No valid segments for user {userID}, skipping.")

    Xtrain = np.vstack(Xtrain).astype(np.float64)
    labels = np.array(labels, dtype=int)

    print(f"[CHECK] Final Xtrain size: {Xtrain.shape[0]} x {Xtrain.shape[1]}")
    print(f"[CHECK] Number of NaN entries: {np.isnan(Xtrain).sum()}")

    if Xtrain.shape[0] == 0:
        raise RuntimeError("Xtrain is empty! Initialization failed.")

    # 标准化训练数据 Standardize training data before training the model
    Xtrain_mean = Xtrain.mean(axis=0)
    Xtrain_std = Xtrain.std(axis=0)
    Xtrain_std[Xtrain_std == 0] = 1.0
    Xtrain = (Xtrain - Xtrain_mean) / Xtrain_std

    # MATLAB: fitcecoc(Xtrain, labels)
    SVMModel = OneVsRestClassifier(SVC(kernel="rbf", gamma="scale"))
    SVMModel.fit(Xtrain, labels)

    db["users"] = [{"id": i+1} for i in range(len(mima))]
    return db


def Train_data(ampDir, Fs_target, band, leadIdx, winSec, db):
    # 测试阶段 (使用提取的特征识别用户)  Testing phase (identify users using extracted features)

    heaFiles = [f for f in os.listdir(ampDir) if f.endswith(".hea")]
    recNames = sorted(list({os.path.splitext(f)[0] for f in heaFiles}))
    groups = {}

    for rec in recNames:
        m = re.match(r"^(.*?_\d+)_\d+$", rec)
        if not m:
            continue
        key = m.group(1)
        groups.setdefault(key, []).append(rec)

    keys = list(groups.keys())
    Xtest = []
    trueLabels = []
    userID = 1

    global Xtrain_mean, Xtrain_std, SVMModel, mima

    for n in range(0, len(keys)):
        recSet = groups[keys[n]]
        if len(recSet) == 0:
            continue

        # 只取每个人最后一条记录 only take the last record for each person。
        rec = recSet[-1]
        ecg, Fs0 = loadAmpECG_manual(ampDir, rec, leadIdx)
        if Fs0 != Fs_target:
            ecg = resample_to(ecg, Fs0, Fs_target)
        ecg = denoiseECG(ecg, Fs_target)

        # 提取最后20秒的特征 extract features from the last 20 seconds
        step = int(Fs_target * winSec)
        if len(ecg) > step:
            ecg_seg = ecg[-step:]
        else:
            ecg_seg = ecg[:min(len(ecg), step)]
            if ecg_seg.size < step:
                ecg_seg = np.pad(ecg_seg, (0, step - ecg_seg.size))

        feat = extractFeatures(ecg_seg, Fs_target)
        vec = feat["vec"]
        if np.any(np.isnan(vec)):
            print(f"[WARN] Test sample for user {userID} has NaN, skipping.")
            continue

        Xtest.append(vec)
        trueLabels.append(userID)
        userID += 1

    if len(Xtest) == 0:
        print("[WARN] Xtest is empty; no predictions can be made.")
        return

    Xtest = np.vstack(Xtest).astype(np.float64)

    # 标准化测试特征 standardize test features
    Xtest = (Xtest - Xtrain_mean) / Xtrain_std

    predictedLabels = SVMModel.predict(Xtest)

    # 打印结果 print results。
    for idx, (t, p) in enumerate(zip(trueLabels, predictedLabels), start=1):
        p = int(p)
        pwd = mima[p-1] if (mima is not None and 1 <= p <= len(mima)) else "<out-of-range>"
        print(f"[{idx:2d}] Known user #{t} -> Predicted user #{p}   pwd={pwd}")


def main():
    # =================== 全局参数 global parameters。===================
    Fs_target = 500
    winSec  = 20
    band      = (0.5, 40)
    batchN    = 4
    margin    = -0.02
    leadIdx   = 1
    ampDir    = "./amp_"
    testDir   = "./test"
    dbFile    = "ecg_db.mat"

    print("[INFO] ECG Auth System started")

    # =================== 0) 载入 / 新建数据库 Load / Create Database===================
    db = {"users": []}
    print("[INFO] Created new empty database")

    # =================== 1) AMP 预初始化 AMP pre-initialization ===================
    if (len(db["users"]) == 0) and os.path.isdir(ampDir):
        print("[INIT] Building initial users from AMP ...")
        db = initUsersFromAMP(ampDir, Fs_target, band, leadIdx, winSec, db)
        Train_data(ampDir, Fs_target, band, leadIdx, winSec, db)
        print(f"[INIT] Users after AMP init: {len(db['users'])}")
    else:
        print(f"[WARN] AMP dir not found or db not empty: ampDir={ampDir}")


if __name__ == "__main__":
    main()