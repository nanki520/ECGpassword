# ======================================================================
#  ECG_Auth_System.py      rev-I  (2025-04-24)
#  FINAL_SVM  (raw-window multiclass SVM)
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
# 降噪 Denoising
def denoiseECG(x, Fs):
    d_low, d_high = 0.5, 40.0
    b, a = butter(4, [d_low/(Fs/2), d_high/(Fs/2)], btype="band")
    y = filtfilt(b, a, x).astype(np.float64)
    # y = (y - np.mean(y)) / np.std(y)   # MATLAB commented out
    return y


def resample_to(x, fs0, fs1):
    """Like MATLAB resample(ecg, Fs_target, Fs0) but robust via rational resample_poly."""
    if fs0 == fs1:
        return np.asarray(x, dtype=np.float64)
    from fractions import Fraction
    frac = Fraction(fs1, fs0).limit_denominator(2000)
    return resample_poly(np.asarray(x, dtype=np.float64), frac.numerator, frac.denominator).astype(np.float64)


# —— 提取基本特征 extract basic features—— %
def extractFeatures(ecg, Fs):
    ecg = np.asarray(ecg, dtype=np.float64)

    # —— 时域特征 time-domain features—— %
    mph = np.mean(ecg) + 0.5*np.std(ecg)
    mpd = int(round(0.25*Fs))
    loc, _ = find_peaks(ecg, height=mph, distance=max(1, mpd))

    RR = np.diff(loc) / Fs if loc.size >= 2 else np.array([])
    if RR.size < 3:
        RR = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

    meanRR = np.nanmean(RR)
    SDNN = np.nanstd(RR)
    RMSSD = np.sqrt(np.nanmean(np.diff(RR)**2)) if RR.size >= 2 else np.nan

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
        shape = float(np.nanmean(ccs)) if len(ccs) else np.nan
    else:
        shape = np.nan

    # —— 频域HRV特征 frequency-domain HRV features—— %
    f, pxx = welch(ecg, fs=Fs, window="hamming", nperseg=512, noverlap=256, nfft=1024, scaling="density")

    def bandpower(bnd):
        lo, hi = bnd
        m = (f >= lo) & (f <= hi)
        if not np.any(m):
            return 0.0
        return float(np.trapz(pxx[m], f[m]))

    LF = bandpower((0.04, 0.15))
    HF = bandpower((0.15, 0.40))
    LFHF = LF / max(HF, np.finfo(np.float64).eps)

    vec = np.round(np.array([meanRR, SDNN, RMSSD, shape, LF, HF, LFHF], dtype=np.float64), 3)
    return {
        "meanRR": meanRR, "SDNN": SDNN, "RMSSD": RMSSD,
        "shape": shape, "LF": LF, "HF": HF, "LFHF": LFHF,
        "vec": vec
    }


# —— generate a hashed password —— %
def generatePassword(vec):
    s = "_".join([f"{float(x):.3f}" for x in np.asarray(vec).ravel().tolist()])
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest().lower()


# ---------- 文件读取 file reading ----------
def loadAmpECG_manual(folder, rec, lead):
    """
    Mirrors MATLAB loadAmpECG_manual.
    Expects: rec.hea and rec.dat in folder.
    """
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

    # MATLAB uses tok{3} and tok{6} (1-based) => tok[2], tok[5] in Python
    gain = float(tok[2])
    baseline = float(tok[5])

    raw = np.fromfile(dat_path, dtype="<i2").astype(np.float64)
    L = (raw.size // nSig) * nSig
    raw = raw[:L]
    sig = raw.reshape(nSig, -1)[lead-1, :]

    if gain != 0:
        sig = (sig - baseline) / gain

    # Wavelet: coif4, level=7; set A7=0 then reconstruct
    wname = "coif4"
    level = 7
    coeffs = pywt.wavedec(sig, wname, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # A7=0
    sig = pywt.waverec(coeffs, wname)

    return sig.astype(np.float64), Fs


# ---------- AMP 初始化 pre-initialize AMP----------
def initUsersFromAMP(ampDir, Fs_target, band, leadIdx, winSec, db):
    heaFiles = [f for f in os.listdir(ampDir) if f.endswith(".hea")]
    recNames = sorted(list({os.path.splitext(f)[0] for f in heaFiles}))
    groups = {}

    # group by ^(.*?_\d+)_\d+$
    for rec in recNames:
        m = re.match(r"^(.*?_\d+)_\d+$", rec)
        if not m:
            continue
        key = m.group(1)
        groups.setdefault(key, []).append(rec)

    keys = list(groups.keys())

    Xtrain = []
    lable = []  # keep MATLAB spelling
    kk = 1
    kkk = 1
    Xall = []

    global mima, SVMModel
    mima = []

    for n in range(0, max(0, len(keys) - 1)):
        recSet = groups[keys[n]]

        for r in recSet[0:len(recSet)]:
            ecg, Fs0 = loadAmpECG_manual(ampDir, r, leadIdx)
            if Fs0 != Fs_target:
                ecg = resample_to(ecg, Fs0, Fs_target)
            ecg = denoiseECG(ecg, Fs_target)

            winLen = int(Fs_target * winSec)

            # MATLAB: feat = extractFeatures(ecg(1:Fs_target*winSec),Fs_target);
            feat = extractFeatures(ecg[0:winLen], Fs_target)
            traindata = ecg[0:winLen]

            Xall.append(feat["vec"])

            if len(ecg) > 100000:
                step = winLen
                for start in range(0, len(ecg) - step, step):
                    lable.append(kkk)
                    kk += 1
                    Xtrain.append(ecg[start:start + step])
            else:
                lable.append(kkk)
                kk += 1
                Xtrain.append(traindata)

        # MATLAB: mima(kkk,:) = generatePassword(mean(Xall,1));
        mean_feat = np.mean(np.vstack(Xall), axis=0) if len(Xall) else np.zeros((7,))
        mima.append(generatePassword(mean_feat))
        kkk += 1

    # 标准化训练数据 normalize training data
    Xtrain = np.vstack(Xtrain).astype(np.float64)
    mu = Xtrain.mean(axis=0)
    sd = Xtrain.std(axis=0)
    sd[sd == 0] = 1.0
    Xtrain = (Xtrain - mu) / sd

    # MATLAB: SVMModel = fitcecoc(Xtrain, lable);
    # Python close analogue: multiclass wrapper over SVC
    SVMModel = OneVsRestClassifier(SVC(kernel="rbf", gamma="scale"))
    SVMModel.fit(Xtrain, np.array(lable, dtype=int))

    # db placeholder
    db["users"] = [{"id": i+1} for i in range(len(mima))]
    return db


def Train_data(ampDir, Fs_target, band, leadIdx, winSec, db):
    """
    Mirror intent of your Train_data: take last record per user group and classify.
    Note: your MATLAB loop in this section had a syntax error; here we implement the likely intent.
    """
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

    global SVMModel, Xtrain_mean, Xtrain_std, mima

    Xtest = []
    winLen = int(Fs_target * winSec)

    for n in range(0, len(keys)):
        recSet = groups[keys[n]]
        r = recSet[-1]  # Only take the last record.

        ecg, Fs0 = loadAmpECG_manual(ampDir, r, leadIdx)
        if Fs0 != Fs_target:
            ecg = resample_to(ecg, Fs0, Fs_target)
        ecg = denoiseECG(ecg, Fs_target)

        # Use a single window from each record:
        if len(ecg) >= winLen:
            # take last window if long, else first window
            if len(ecg) > 100000:
                start = max(0, len(ecg) - winLen)
                seg = ecg[start:start + winLen]
            else:
                seg = ecg[0:winLen]
        else:
            seg = np.pad(ecg, (0, winLen - len(ecg)))

        Xtest.append(seg)

    Xtest = np.vstack(Xtest).astype(np.float64)

    # 测试数据标准化 Standardize test data
    # (MATLAB did this on Xtrain in Train_data; we keep same behavior here)
    Xtrain_mean = Xtest.mean(axis=0)
    Xtrain_std = Xtest.std(axis=0)
    Xtrain_std[Xtrain_std == 0] = 1.0
    Xtest = (Xtest - Xtrain_mean) / Xtrain_std

    predictedLabels = SVMModel.predict(Xtest)

    for kkkk, uid in enumerate(predictedLabels, start=1):
        uid = int(uid)
        pwd = mima[uid-1] if (mima is not None and 1 <= uid <= len(mima)) else "<out-of-range>"
        print(f"[{kkkk:2d}] {'1':<25}  Known user #{uid}  pwd={pwd}")


def main():
    # =================== 全局参数 global parameters。===================
    Fs_target = 500            # 统一采样率 uniform sampling rate
    winSec  = 10               # 窗口长度 (秒) window length (seconds)。
    band      = (0.5, 40)      # 滤波带宽 filter bandwidth。
    batchN    = 4              # 累积 N 段再增量更新 accumulate N segments before incremental update。
    margin    = -0.02          # 模糊带阈值 fuzzy band threshold。
    leadIdx   = 1              # AMP 取第 1 导联 Use lead I from AMP
    ampDir    = "./amp_"       # 可选 AMP 训练目录
    testDir   = "./test"       # 测试 CSV 目录 test CSV directory
    dbFile    = "ecg_db.mat"   # 数据库文件 database file。 (MATLAB name; unused here)

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
