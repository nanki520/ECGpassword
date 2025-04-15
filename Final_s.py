# ======================================================================
#  ECG_Auth_System.py      rev-I  (2025-04-24)
#  FINAL_S  (per-user SVDD / one-class SVM with pending enrollment)
# ======================================================================

import os, re
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks, welch, resample_poly
from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold


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
    # y = (y - mean(y)) ./ std(y);  % MATLAB commented out
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

    meanRR = np.nanmean(RR)
    SDNN   = np.nanstd(RR)
    RMSSD  = np.sqrt(np.nanmean(np.diff(RR)**2)) if RR.size >= 2 else np.nan

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

    LF   = bandpower((0.04, 0.15))
    HF   = bandpower((0.15, 0.40))
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

    # wavelet: coif4 level 7, set A7=0
    wname = "coif4"
    level = 7
    coeffs = pywt.wavedec(sig, wname, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # A7=0
    sig = pywt.waverec(coeffs, wname)

    return sig.astype(np.float64), Fs


# ---------- 自动调参一类 SVDD automatic hyperparameter tuning for one-class ----------
def tuneSVDD(X):
    # X : n×d 特征矩阵
    X = np.asarray(X, dtype=np.float64)

    if X.shape[0] < 2:
        mdl = OneClassSVM(kernel="rbf", gamma=1.0, nu=0.05)
        mdl.fit(X)
        return mdl

    scales = [0.5, 1, 2]   # MATLAB KernelScale analogue
    nus = [0.01, 0.05, 0.1]

    best = -np.inf
    bestMdl = None

    kfold = min(5, X.shape[0])
    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

    for s in scales:
        for n in nus:
            cv_scores = []
            for tr, te in kf.split(X):
                m = OneClassSVM(kernel="rbf", gamma=s, nu=n)
                m.fit(X[tr])
                cv_scores.append(float(m.decision_function(X[te]).mean()))
            score = float(np.mean(cv_scores))
            if score > best:
                best = score
                bestMdl = OneClassSVM(kernel="rbf", gamma=s, nu=n)
                bestMdl.fit(X)

    return bestMdl


def retrainUser(db, k):
    allX = np.vstack([db["users"][k]["X"], db["users"][k]["buf"]]) if db["users"][k]["buf"].size else db["users"][k]["X"]
    mdl = tuneSVDD(allX)

    # ---- Clean up weak support vectors ----
    # MATLAB attempts: use SupportVectors and Alpha; scikit-learn doesn't expose Alpha similarly.
    db["users"][k]["X"] = allX
    db["users"][k]["buf"] = np.zeros((0, allX.shape[1]), dtype=np.float64)
    db["users"][k]["model"] = mdl
    print(f"   Retrained user #{k+1}  samples={allX.shape[0]}  SV={mdl.support_vectors_.shape[0]}")
    return db


# =================  core logic function =================
def authenticateSegment(feats, db, batchN, margin):
    # feats: S×d window feature matrix
    # batchN: 累计文件数门限 accumulated file count threshold
    # margin: SVDD 判定阈值 SVDD decision threshold
    # uid > 0: 已识别用户编号 identified user ID
    # uid == 0: 新用户注册成功，返回新 pwd New user registered successfully, return new pwd

    uid = -2
    pwd = ""

    nU = len(db["users"])
    x = np.mean(feats, axis=0)  # 用本文件的平均特征进行打分 Score using the average features of the current file

    # —— 1) 对已有用户判分 Scoring for existing users—— %%
    if nU > 0:
        scores = np.zeros((nU,), dtype=np.float64)
        for k in range(nU):
            mu = db["users"][k]["mu"]
            sigma = db["users"][k]["sigma"]
            x_norm = (x - mu) / sigma  # Normalize x_raw using each user's own mu and sigma
            sc = float(db["users"][k]["model"].decision_function(x_norm.reshape(1, -1))[0])
            scores[k] = sc  # decision value of the one-class SVM

        bestId = int(np.argmax(scores))
        bestScore = float(scores[bestId])

        if bestScore >= margin:
            uid = bestId + 1
            pwd = db["users"][bestId]["pwd"]

            db["users"][bestId]["buf"] = np.vstack([db["users"][bestId]["buf"], feats])

            # 如果 buf 样本数达到 batchN*窗口数，触发重训练
            if db["users"][bestId]["buf"].shape[0] >= batchN * feats.shape[0]:
                db = retrainUser(db, bestId)
            return uid, pwd, db

    # —— 2) 新用户注册逻辑 —— %%
    if db.get("pending", None) is None or (isinstance(db["pending"], np.ndarray) and db["pending"].size == 0):
        db["pending"] = feats.copy()
        return uid, pwd, db
    else:
        db["pending"] = np.vstack([db["pending"], feats])

    # 计算已累计了多少文件：pending 行数 / 每文件窗口数
    numFiles = db["pending"].shape[0] / feats.shape[0]
    if numFiles < batchN:
        return uid, pwd, db

    # 累够 batchN 份，真正注册
    new = {}
    new["Xraw"] = db["pending"]
    new["mu"] = np.mean(new["Xraw"], axis=0)
    new["sigma"] = np.std(new["Xraw"], axis=0) + np.finfo(np.float64).eps
    Xnorm = (new["Xraw"] - new["mu"]) / new["sigma"]  # ★★★ 归一化 ★★★

    new["X"] = Xnorm
    new["buf"] = np.zeros((0, Xnorm.shape[1]), dtype=np.float64)
    new["model"] = tuneSVDD(new["X"])
    new["pwd"] = generatePassword(np.mean(new["X"], axis=0))

    db["users"].append(new)
    uid = 0
    pwd = new["pwd"]
    db["pending"] = None
    return uid, pwd, db


# ---------- AMP 分组初始化 + 建立模型 ----------
def initUsersFromAMP(ampDir, Fs_target, band, leadIdx, db):
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

    for n in range(0, max(0, len(keys) - 1)):
        recSet = groups[keys[n]]
        Xtrain = []

        for r in recSet[0:4]:  # 对于这个用户的前4条记录，逐条处理
            ecg, Fs0 = loadAmpECG_manual(ampDir, r, leadIdx)
            if Fs0 != Fs_target:
                ecg = resample_to(ecg, Fs0, Fs_target)
            ecg = denoiseECG(ecg, Fs_target)

            if len(ecg) > 100000:
                step = int(Fs_target * 20)
                for start in range(0, len(ecg) - step, step):
                    ecg_seg = ecg[start:start + step]
                    feat = extractFeatures(ecg_seg, Fs_target)
                    Xtrain.append(feat["vec"])
            else:
                step = int(Fs_target * 20)
                ecg_seg = ecg[:min(len(ecg), step)]
                if ecg_seg.size < step:
                    ecg_seg = np.pad(ecg_seg, (0, step - ecg_seg.size))
                feat = extractFeatures(ecg_seg, Fs_target)
                Xtrain.append(feat["vec"])

        if len(Xtrain) == 0:
            print(f"Warning: User group {keys[n]} has no valid ECG segments, skipped.")
            continue

        Xtrain = np.vstack(Xtrain)

        # ====== 标准化特征 ======
        mu = np.mean(Xtrain, axis=0)
        sigma = np.std(Xtrain, axis=0) + np.finfo(np.float64).eps
        Xtrain_norm = (Xtrain - mu) / sigma

        # ====== 训练 SVDD ======
        new = {}
        new["model"] = tuneSVDD(Xtrain_norm)
        new["mu"] = mu
        new["sigma"] = sigma
        new["X"] = Xtrain_norm
        new["Xraw"] = np.zeros((0, Xtrain.shape[1]), dtype=np.float64)  # ★补上空Xraw，保证一致
        new["buf"] = np.zeros((0, Xtrain.shape[1]), dtype=np.float64)
        new["pwd"] = generatePassword(np.mean(Xtrain_norm, axis=0))

        db["users"].append(new)

        print(f"   User {keys[n]} -> user #{len(db['users'])} (segments={len(recSet)})")

    return db


def Train_data(ampDir, Fs_target, band, leadIdx, db):
    # Demo scoring over AMP “last record” per group (as in your MATLAB code)
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

    for n in range(0, len(keys)):
        recSet = groups[keys[n]]

        if len(recSet) < 5:
            print(f"  Warning: User {keys[n]} has less than 5 records. Skipped.")
            continue

        r = recSet[-1]
        ecg, Fs0 = loadAmpECG_manual(ampDir, r, leadIdx)
        if Fs0 != Fs_target:
            ecg = resample_to(ecg, Fs0, Fs_target)
        ecg = denoiseECG(ecg, Fs_target)

        step = int(Fs_target * 20)
        if len(ecg) > step:
            ecg_seg = ecg[-step:]
        else:
            ecg_seg = ecg[:min(len(ecg), step)]
            if ecg_seg.size < step:
                ecg_seg = np.pad(ecg_seg, (0, step - ecg_seg.size))

        feat = extractFeatures(ecg_seg, Fs_target)
        Xall = feat["vec"].reshape(1, -1)

        nU = len(db["users"])
        if nU > 0:
            scores = np.zeros((nU,), dtype=np.float64)
            for k in range(nU):
                mu = db["users"][k]["mu"]
                sigma = db["users"][k]["sigma"]
                X_norm = (Xall - mu) / sigma
                sc = float(db["users"][k]["model"].decision_function(X_norm)[0])
                scores[k] = sc

            bestId = int(np.argmax(scores))
            bestScore = float(scores[bestId])
            uid = bestId + 1
            pwd = db["users"][bestId]["pwd"]
            print(f"Test sample matched to user #{uid}, score = {bestScore:.4f}, pwd={pwd}")


def main():
    print("[INFO] ECG Auth System started")

    # =================== 全局参数 ===================
    Fs_target = 500
    winSec  = 20
    band      = (0.5, 40)
    batchN    = 3
    margin    = -0.01
    leadIdx   = 1
    ampDir    = "./amp_"
    testDir   = "./test"
    dbFile    = "ecg_db.mat"  # MATLAB name; we keep structure in-memory only

    # =================== 0) 载入 / 新建数据库 Load / Create Database ===================
    db = {"users": [], "pending": None}
    print("[INFO] Created new empty database")

    # =================== 1) AMP 预初始化 pre-initialize AMP ===================
    if (len(db["users"]) == 0) and os.path.isdir(ampDir):
        print("[INIT] Building initial users from AMP ...")
        db = initUsersFromAMP(ampDir, Fs_target, band, leadIdx, db)
        Train_data(ampDir, Fs_target, band, leadIdx, db)
        print(f"[INIT] Users after AMP init: {len(db['users'])}")
    else:
        print(f"[WARN] AMP dir not found or db not empty: ampDir={ampDir}")

    # ================== 主程序（处理 dat/hea 文件） ==================
    if not os.path.isdir(testDir):
        raise RuntimeError(f'testDir not found: "{testDir}"')

    datList = [f for f in os.listdir(testDir) if f.endswith(".dat")]
    datList.sort()

    if len(datList) == 0:
        raise RuntimeError(f'No .dat files found in "{testDir}"')

    winLen = int(winSec * Fs_target)  # 每窗采样点数 number of samples per window

    for k, datname in enumerate(datList, start=1):
        fname = os.path.splitext(datname)[0]
        sig, Fs = loadAmpECG_manual(testDir, fname, leadIdx)

        if Fs != Fs_target:
            sig = resample_to(sig, Fs, Fs_target)

        sig = denoiseECG(sig, Fs_target)

        # —— 分段，提取多窗口特征 Segment the signal and extract multi-window features —— %
        S = len(sig) // winLen
        if S == 0:
            print(f"[{k:2d}] {fname:<20} too short (<{winSec} s), skipped")
            continue

        feats = np.zeros((S, 7), dtype=np.float64)
        for s in range(S):
            seg = sig[s*winLen:(s+1)*winLen]
            featStruct = extractFeatures(seg, Fs_target)
            feats[s, :] = featStruct["vec"]

        # —— Perform authentication after file feature extraction is complete —— %
        uid, pwd, db = authenticateSegment(feats, db, batchN, margin)

        # —— 输出认证结果 Output authentication results —— %
        if uid > 0:
            print(f"[{k:2d}] {fname:<20}  Known user #{uid}  pwd={pwd}")
        elif uid == 0:
            print(f"[{k:2d}] {fname:<20}  New user       pwd={pwd}")
        else:
            print(f"[{k:2d}] {fname:<20}  Uncertain, need more segments")

    print(f"[INFO] Done. Users in DB: {len(db['users'])}")


if __name__ == "__main__":
    main()
