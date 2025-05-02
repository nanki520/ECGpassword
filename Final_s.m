% ======================================================================
%  ECG_Auth_System.m      rev-I  (2025-04-24)
% ======================================================================
clc; clear; close all;

%% =================== 全局参数 ===================
Fs_target = 500;            % 统一采样率 uniform sampling rate
winSec  = 20;             % 窗口长度 (秒)window length (seconds)。
band      = [0.5 40];       % 滤波带宽 filter bandwidth。
batchN    = 3;              % 累积 N 段再增量更新 accumulate N segments before incremental update。
margin    = -0.01;           %  模糊带阈值  fuzzy band threshold。
leadIdx   = 1;              % AMP 取第 1 导联 Use lead I from AMP 
ampDir    = './amp_';        % 可选 AMP 训练目录
testDir   = './test';       % 测试 CSV 目录 test CSV directory
dbFile    = 'ecg_db.mat';   % 数据库文件 database file。


fprintf('[INFO] ECG Auth System started  %s\n', datestr(now,31));

%% =================== 0) 载入 / 新建数据库Load / Create Database ===================
if isfile(dbFile)&&0
    s  = load(dbFile,"db"); db = s.db;
    fprintf('[INFO] Loaded database (%d users)\n', numel(db.users));
else
    db.users = struct([]);
    save(dbFile,"db","-mat");
    fprintf('[INFO] Created new empty database\n');
end

%% =================== 1) AMP 预初始化 pre-initialize AMP ===================
if isempty(db.users) && isfolder(ampDir)
    fprintf('[INIT] Building initial users from AMP ...\n');
    db = initUsersFromAMP(ampDir, Fs_target, band, leadIdx, db);
    Train_data(ampDir, Fs_target, band, leadIdx, db);
    save(dbFile,"db","-mat");
    fprintf('[INIT] Users after AMP init: %d\n', numel(db.users));
end


% ================== 主程序（处理 dat/hea 文件） ==================

datList = dir(fullfile(testDir, '*.dat'));
if isempty(datList)
    error('No .dat files found in "%s"', testDir);
end

winLen = winSec * Fs_target;  % 每窗采样点数 number of samples per window

for k = 1:numel(datList)
    % —— 载入一条ECG Load one ECG record —— %
    fname = erase(datList(k).name, '.dat');  %  Remove .dat extension to get the record name
    [sig, Fs] = loadAmpECG_manual(testDir, fname, leadIdx);

    if Fs ~= Fs_target
        sig = resample(sig, Fs_target, Fs);  % 重采样到目标采样率 Resample to the target sampling rate
    end

    sig = denoiseECG(sig, Fs_target);  % 去噪 denoise。

    % —— 分段，提取多窗口特征 Segment the signal and extract multi-window features —— %
    S = floor(numel(sig) / winLen);
    if S == 0
        warning('%s too short (<%d s), skipped', fname, winSec);
        continue;
    end

    feats = zeros(S, 7); 
    for s = 1:S
        seg = sig((s-1)*winLen+1 : s*winLen);
        featStruct = extractFeatures(seg, Fs_target);
        feats(s,:) = featStruct.vec;
    end

    % ——  Perform authentication after file feature extraction is complete —— %
    [uid, pwd, db] = authenticateSegment(feats, db, batchN, margin);

    % —— 输出认证结果 Output authentication results —— %
    if uid > 0
        fprintf('[%2d] %-20s  Known user #%d  pwd=%s\n', ...
                k, fname, uid, pwd);
    elseif uid == 0
        fprintf('[%2d] %-20s  New user       pwd=%s\n', ...
                k, fname, pwd);
    else
        fprintf('[%2d] %-20s  Uncertain, need more segments\n', ...
                k, fname);
    end
end

% —— 保存数据库 Save the database—— %
save(dbFile, "db", "-mat");
fprintf('[INFO] Done. Users in DB: %d\n', numel(db.users));



%% =================  core logic function =================

function [uid, pwd, db] = authenticateSegment(feats, db, batchN, margin)
% feats: S×d window feature matrix 
% batchN: 累计文件数门限 accumulated file count threshold
% margin: SVDD 判定阈值  SVDD decision threshold
%   uid > 0: 已识别用户编号 identified user ID
%   uid == 0: 新用户注册成功，返回新 pwd  New user registered successfully, return new pwd

uid = -2;  
pwd = "";

nU = numel(db.users);
x   = mean(feats,1);  % 用本文件的平均特征进行打分 Score using the average features of the current file


%% —— 1) 对已有用户判分 Scoring for existing users—— %%
if nU > 0
    scores = zeros(nU,1);
    for k = 1:nU
        mu = db.users(k).mu;
        sigma = db.users(k).sigma;
        x = (x - mu) ./ sigma;  % Normalize x_raw using each user's own mu and sigma

        [~, sc]   = predict(db.users(k).model, x);
        scores(k) = sc;  %  decision value of the one-class SVM。
    end
    [bestScore, bestId] = max(scores);

    if bestScore >= margin
        % 判定为 bestId  Classified as bestId

        uid = bestId;
        pwd = db.users(uid).pwd;
        % 追加到该用户的 buf（所有窗口特征）
        db.users(uid).buf = [db.users(uid).buf; feats];
        % 如果 buf 样本数达到 batchN*窗口数，触发重训练
        if size(db.users(uid).buf,1) >= batchN * size(feats,1)
            db = retrainUser(db, uid);
        end
        return;
    %elseif bestScore >= -margin
        % 落入模糊带：既不识别也不注册
        %return;
    end
end

%% —— 2) 新用户注册逻辑 —— %%
% 用 db.pending 存储“待注册”用户的窗口特征
if ~isfield(db, 'pending') || isempty(db.pending)
    % 第一次出现未知用户，就创建 pending
    db.pending = feats;
    return;
else
    % 追加新的文件级特征
    db.pending = [db.pending; feats];
end

% 计算已累计了多少文件：pending 行数 除以 每文件窗口数
numFiles = size(db.pending,1) / size(feats,1);
if numFiles < batchN
    % 还没累够 batchN 份，继续等待
    return;
end

% 累够 batchN 份，真正注册
new.Xraw  = db.pending;              % 保存原始特征
new.mu    = mean(new.Xraw, 1);        % 求自己的mu
new.sigma = std(new.Xraw, 0, 1) + eps; % 求自己的sigma，防止除0
Xnorm     = (new.Xraw - new.mu) ./ new.sigma;  % ★★★ 归一化 ★★★

new.X     = Xnorm  ;            % 用所有累积窗口特征做训练集
new.buf   = [];                    % 新注册时 buf 置空
new.model = tuneSVDD(new.X);       % 训练 SVDD 模型
new.pwd   = generatePassword(mean(new.X,1));

% 添加到用户库
db.users(end+1) = new;
uid = 0;
pwd = new.pwd;

% 清理 pending
db.pending = [];
end



function db = retrainUser(db,k)
allX = [db.users(k).X; db.users(k).buf];
mdl  = tuneSVDD(allX);
% ---- 清理弱支持向量 Clean up weak support vectors ----
sv  = mdl.SupportVectors;
alp = abs(mdl.Alpha) > 1e-4;
mdl = tuneSVDD(sv(alp,:));
db.users(k).X     = allX;
db.users(k).buf   = [];
db.users(k).model = mdl;
fprintf('   Retrained user #%d  samples=%d  SV=%d\n', ...
    k, size(allX,1), size(mdl.SupportVectors,1));
end

%% ---------- 自动调参一类 SVDD automatic hyperparameter tuning for one-class ----------

function mdl = tuneSVDD(X)
% X : n×d 特征矩阵
Y = ones(size(X,1),1);

if size(X,1) < 2                      % 仅 1 条样本 → 无法做 CV  Only 1 sample → cannot perform CV

    mdl = fitcsvm(X,Y,'KernelFunction','rbf', ...
        'KernelScale',1,'Nu',0.05,'Standardize',true);
    return
end

scales = [0.5 1 2]; nus = [0.01 0.05 0.1];
best = inf; bestMdl = [];
cv = cvpartition(Y,'KFold',min(5,size(X,1)));
for s = scales
    for n = nus
        m = fitcsvm(X,Y,'KernelFunction','rbf', ...
            'KernelScale',s,'Nu',n,'Standardize',true);
        loss = kfoldLoss(crossval(m,'CVPartition',cv));
        if loss < best, best = loss; bestMdl = m; end
    end
end
mdl = bestMdl;
end



%% ---------- AMP 分组初始化 +建立模型 ----------
function db = initUsersFromAMP(ampDir, Fs_target, band, leadIdx, db)

% 列出所有 hea 文件
heaFiles = dir(fullfile(ampDir,'*.hea'));
recNames = unique( erase({heaFiles.name}, '.hea') );
groups   = containers.Map;

% 按用户分组
for f = recNames
    m = regexp(f{1},'^(.*?_\d+)_\d+$','tokens'); 
    if isempty(m), continue; end
    key = m{1}{1};
    if isKey(groups,key)
        tmp = groups(key); 
        tmp{end+1} = f{1}; 
        groups(key) = tmp;
    else
        groups(key) = {f{1}};
    end
end

keys = groups.keys;

%遍历每一个用户组
for n = 1:numel(keys)-1 
    recSet = groups(keys{n}); 
    Xtrain = [];

    for r = recSet(1:4) %对于这个用户的前4条记录，逐条处理。

        [ecg,Fs0] = loadAmpECG_manual(ampDir,r{1},leadIdx);
        if Fs0~=Fs_target
            ecg = resample(ecg,Fs_target,Fs0); 
        end
        ecg = denoiseECG(ecg,Fs_target);

        % 分割 ECG，提取特征
        if(length(ecg)>100000)
            for kkkkk= 1:Fs_target*20:(length(ecg)-Fs_target*20)
                ecg_seg = ecg(kkkkk:(kkkkk+Fs_target*20-1));
                feat = extractFeatures(ecg_seg, Fs_target);
                Xtrain = [Xtrain; feat.vec];
            end
        else
            ecg_seg = ecg(1:min(length(ecg), Fs_target*20));
            feat = extractFeatures(ecg_seg, Fs_target);
            Xtrain = [Xtrain; feat.vec];
        end
    end

    % —— 检查有没有特征 —— %
    if isempty(Xtrain)
        warning('User group %s has no valid ECG segments, skipped.', keys{n});
        continue;
    end

      % ====== 标准化特征 ======
    mu = mean(Xtrain, 1);
    sigma = std(Xtrain, 0, 1) + eps; % 防止除以0
    Xtrain_norm = (Xtrain - mu) ./ sigma;

     % ====== 训练 SVDD ======
    new.model = tuneSVDD(Xtrain_norm);
    new.mu    = mu;
    new.sigma = sigma;
    new.X     = Xtrain_norm;  % 保存标准化后的数据
    new.Xraw = [];   % ★补上空Xraw，保证一致
    new.buf   = [];
    new.pwd   = generatePassword(mean(Xtrain_norm,1));

    % 保存到 db
    if isempty(db.users)
        db.users = new; 
    else 
        db.users(end+1)=new; 
    end

    fprintf('   User %s -> user #%d (segments=%d)\n', ...
        keys{n}, numel(db.users), numel(recSet));
end
end


function  Train_data(ampDir, Fs_target, band, leadIdx, db)

heaFiles = dir(fullfile(ampDir,'*.hea'));
recNames = unique( erase({heaFiles.name}, '.hea') );
groups   = containers.Map;

for f = recNames
    m = regexp(f{1},'^(.*?_\d+)_\d+$','tokens'); 
    if isempty(m), continue; end
    key = m{1}{1};
    if isKey(groups,key)
        tmp = groups(key); 
        tmp{end+1} = f{1}; 
        groups(key) = tmp;
    else
        groups(key) = {f{1}};
    end
end

keys = groups.keys;
uid = -2;   % 默认未识别
pwd = "";   % 默认空密码

% ——  遍历每个测试用户组 —— %
for n = 1:numel(keys)
    recSet = groups(keys{n}); 
    Xall = [];

    if numel(recSet) < 5
        fprintf('  Warning: User %s has less than 5 records. Skipped.\n', keys{n});
        continue;  % 如果不足5条，跳过
    end

    % —— 4. 取最后一条记录 —— %
    r = recSet(end);
    [ecg,Fs0] = loadAmpECG_manual(ampDir,r{1},leadIdx);
    if Fs0~=Fs_target
        ecg = resample(ecg,Fs_target,Fs0); 
    end
    ecg = denoiseECG(ecg,Fs_target);

    % —— 提取最后20秒的特征 —— %
    if length(ecg) > Fs_target * 20
        ecg_seg = ecg(end - Fs_target*20 + 1 : end);
    else
        ecg_seg = ecg(1:min(end, Fs_target*20));
    end

    feat = extractFeatures(ecg_seg,Fs_target);
    Xall = [Xall; feat.vec];% 单条特征向量
    
    nU = numel(db.users);

    %% —— 1) 对已有用户判分 —— %%
    if nU > 0
        scores = zeros(nU,1);
        for k = 1:nU
            mu    = db.users(k).mu;
            sigma = db.users(k).sigma;
            X_norm = (Xall - mu) ./ sigma;  % 按该用户标准化
            [~, sc]   =  predict(db.users(k).model, Xall);
            scores(k) = sc;  % 一类 SVM 的 decision value
        end
        
        % —— 7. 选择得分最高的用户 —— %
        [bestScore, bestId] = max(scores);
        uid = bestId;% 判定为 bestId
        scores(uid)
        pwd = db.users(uid).pwd;
        % 追加到该用户的 buf（所有窗口特征）
        fprintf('Test sample matched to user #%d, score = %.4f\n', uid, bestScore);
    end
end
end

%% ---------- 文件读取 ----------
function [sig, Fs] = loadAmpECG_manual(folder, rec, lead)
txt = fileread(fullfile(folder,[rec '.hea']));
lines = regexp(txt,'\r?\n','split'); lines = lines(~cellfun('isempty',lines));
lines = lines(~startsWith(lines,'#'));
hdr   = regexp(lines{1},'\s+','split');
nSig  = str2double(hdr{2}); 
Fs = str2double(hdr{3});%Sampling rate 采样率
rows  = lines(2:end); 
rows = rows(~contains(rows,'filtered','IgnoreCase',true));
lead  = min(lead,numel(rows));
tok   = regexp(rows{lead},'\s+','split');
gain  = str2double(tok{3}); baseline = str2double(tok{6});
fid = fopen(fullfile(folder,[rec '.dat']),'r','ieee-le');
raw = fread(fid, inf, 'int16=>double'); fclose(fid);
sig = reshape(raw, nSig, []); 
sig = sig(lead,:);
if gain ~= 0, sig = (sig - baseline)/gain; end
wname = 'coif4';    % 小波类型 wavelet type
level = 7;          % 分解尺度 decomposition scale
[C, L] = wavedec(sig, level, wname); % 多尺度分解multi-scale decomposition

a7_start = 1;                  % A7在C中的起始位置 Starting position of A7 in C。
a7_end = L(1);                 % A7在C中的结束位置 Ending position of A7 in C。
C_zeroA7 = C;                  % 复制系数 copy coefficients
C_zeroA7(a7_start:a7_end) = 0; % 将A7部分置零 Set the A7 component to zero 

%% 4. 重构信号（A7=0）
sig = waverec(C_zeroA7, L, wname); % 重构信号 reconstructed signal。
end

function [sig, Fs] = loadCsvECG(fname, targetFs)
data = readmatrix(fname,'NumHeaderLines',13,'OutputType','double');
if isvector(data), raw = double(data(:));
else, raw = double(data(:,1)); end
sig = resample(raw.', targetFs, targetFs); Fs = targetFs;
end

%% =================  辅助函数   =================
% 降噪 Denoising
function y = denoiseECG(x, Fs)
d = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',40, ...
    'SampleRate',Fs);
y = filtfilt(d, x);
%y = (y - mean(y)) ./ std(y);
end

% —— 提取基本特征 extract basic features—— %
function feat = extractFeatures(ecg, Fs)

% —— 时域特征 time-domain features—— %
[~,loc] = findpeaks(ecg, ...
    'MinPeakHeight', mean(ecg) + 0.5*std(ecg), ...
    'MinPeakDistance', round(0.25*Fs));  % 检测心电信号中的峰值 Detect peaks in ECG signals.
RR = diff(loc) / Fs;%RR间期 RR interval
if numel(RR) < 3, RR = nan(1,3); end
% 平均RR间期 平均每次心跳的间隔，代表基础心率 mean RR interval the baseline heart rate.
feat.meanRR = mean(RR, 'omitnan');
% RR间期标准差 表示心率波动大小（整体变异性）The standard deviation of RR intervals the magnitude of heart rate fluctuations (overall variability).
feat.SDNN   = std( RR, 'omitnan');
% 连续RR差分的均方根 主要反映短期心率变化  root mean square of successive differences (RMSSD) of RR intervals short-term heart rate variability.
feat.RMSSD  = rms( diff(RR), 'omitnan');

% —— 波形相似度 tsimilarity between waveforms—— %
beatLen = round(0.7 * Fs);
tmpl    = ecg( max(1,loc(1)-beatLen/2) : min(end,loc(1)+beatLen/2-1) );
nCmp    = min(8, numel(loc)-1);
ccs     = nan(1, nCmp);
for k = 1:nCmp
    seg = ecg( max(1,loc(k+1)-beatLen/2) : min(end,loc(k+1)+beatLen/2-1) );
    L   = min(numel(tmpl), numel(seg));
    if L >= 10
        R = corrcoef(tmpl(1:L), seg(1:L));
        ccs(k) = R(1,2);
    end
end
feat.shape = mean(ccs, 'omitnan');

% —— 频域HRV特征 frequency-domain HRV features—— %
% 计算功率谱密度 compute the power spectral density (PSD)
[pxx, f] = pwelch(ecg, hamming(512), 256, 1024, Fs);
% LF（0.04–0.15 Hz）功率
feat.LF   = bandpower(pxx, f, [0.04 0.15], 'psd');
% HF（0.15–0.40 Hz）功率
feat.HF   = bandpower(pxx, f, [0.15 0.40], 'psd');
% LF/HF 比值
feat.LFHF = feat.LF / max(feat.HF, eps);

% —— 返回行向量 return a row vector—— %
names = fieldnames(feat);
v     = cellfun(@(f) round(feat.(f), 3), names);
feat.vec = v(:)';
end

% —— generate a hashed password —— %
function pwd = generatePassword(vec)
s  = sprintf('%.3f_',vec);
s(end)=[];
md = java.security.MessageDigest.getInstance('SHA-256');
h  = md.digest(uint8(s));
pwd = lower(reshape(dec2hex(typecast(h,'uint8'))',1,[]));
end
