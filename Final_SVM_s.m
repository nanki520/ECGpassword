% ======================================================================
%  ECG_Auth_System.m      rev-I  (2025-04-24)
% ======================================================================
clc; clear; close all;

%% =================== 全局参数 global parameters。===================
Fs_target = 500;            % 统一采样率 uniform sampling rate
winSec  = 20;             % 窗口长度 (秒)window length (seconds)。
band      = [0.5 40];       % 滤波带宽 filter bandwidth。
batchN    = 4;              % 累积 N 段再增量更新 accumulate N segments before incremental update。
margin    = -0.02;           %  模糊带阈值  fuzzy band threshold。
leadIdx   = 1;              % AMP 取第 1 导联 Use lead I from AMP 
ampDir    = './amp_';        % 可选 AMP 训练目录
testDir   = './test';       % 测试 CSV 目录 test CSV directory
dbFile    = 'ecg_db.mat';   % 数据库文件 database file。


fprintf('[INFO] ECG Auth System started  %s\n', datestr(now,31));

%% =================== 0) 载入 / 新建数据库 Load / Create Database===================
if isfile(dbFile)&&0
    s  = load(dbFile,"db"); db = s.db;
    fprintf('[INFO] Loaded database (%d users)\n', numel(db.users));
else
    db.users = struct([]);
    save(dbFile,"db","-mat");
    fprintf('[INFO] Created new empty database\n');
end

%% =================== 1) AMP 预初始化 AMP pre-initialization ===================
if isempty(db.users) && isfolder(ampDir)
    fprintf('[INIT] Building initial users from AMP ...\n');
    db = initUsersFromAMP(ampDir, Fs_target, band, leadIdx, winSec, db);
    Train_data(ampDir, Fs_target, band, leadIdx, winSec, db);
    save(dbFile,"db","-mat");
    fprintf('[INIT] Users after AMP init: %d\n', numel(db.users));
end




%% ---------- AMP 分组初始化 ----------
function db = initUsersFromAMP(ampDir, Fs_target, band, leadIdx, winSec, db)

heaFiles = dir(fullfile(ampDir,'*.hea'));
recNames = unique( erase({heaFiles.name}, '.hea') );
groups   = containers.Map;

for f = recNames
    m = regexp(f{1},'^(.*?_\d+)_\d+$','tokens'); 
    if isempty(m), continue; end
    key = m{1}{1};
    if isKey(groups,key)
        tmp = groups(key); tmp{end+1} = f{1}; groups(key) = tmp;
    else
        groups(key) = {f{1}};
    end
end

keys = groups.keys;
Xtrain = [];
labels  = [];
userID = 1;
global mima
mima = [];

for n = 1:numel(keys)-1
    recSet = groups(keys{n});
    user_feats = [];  % 每个用户单独积

    for r = recSet(1:(length(recSet)))

        [ecg,Fs0] = loadAmpECG_manual(ampDir,r{1},leadIdx);
        if Fs0~=Fs_target
            ecg = resample(ecg,Fs_target,Fs0);
        end
        ecg = denoiseECG(ecg,Fs_target);
        
        if(length(ecg)>100000)
            for start_idx = 1 : Fs_target*winSec  : (length(ecg) - Fs_target*winSec )
                ecg_seg = ecg(start_idx : start_idx + Fs_target*winSec  - 1);
                feat = extractFeatures(ecg_seg, Fs_target);
                vec = feat.vec;

                % 检查并处理NaN Check and handle NaN values
                if any(isnan(vec))
                    continue;  % 直接跳过这段坏样本
                end

                Xtrain = [Xtrain; feat.vec];
                labels = [labels; userID];
                user_feats = [user_feats; vec];
            end
        else
            ecg_seg = ecg(1:min(length(ecg), Fs_target*winSec ));
            feat = extractFeatures(ecg_seg, Fs_target);
            vec = feat.vec;

            % 检查并处理NaN Check and handle NaN values
            if any(isnan(vec))
                continue;  % 直接跳过这段坏样本
            end

            Xtrain = [Xtrain; feat.vec];
            labels = [labels; userID];
            user_feats = [user_feats; vec];
        end
    end

    % 生成密码 enerate password
    if ~isempty(user_feats)
        mean_feat =  mean(user_feats, 1);
        pwd = generatePassword(mean_feat);
        mima(userID,:) = pwd;
        fprintf('[USER] Registered user %d  (%d segments)\n', userID, size(user_feats,1));
        userID = userID + 1;
    else
        fprintf('[WARN] No valid segments for user %d, skipping.\n', userID);
    end
end

% 检查最终Xtrain Check final Xtrain
fprintf('[CHECK] Final Xtrain size: %d x %d\n', size(Xtrain,1), size(Xtrain,2));
fprintf('[CHECK] Number of NaN entries: %d\n', sum(isnan(Xtrain(:))));

if isempty(Xtrain)
    error('Xtrain is empty! Initialization failed.');
end

% 标准化训练数据 Standardize training data before training the model
global Xtrain_mean Xtrain_std
Xtrain_mean = mean(Xtrain, 1);
Xtrain_std = std(Xtrain, [], 1);
Xtrain_std(Xtrain_std == 0) = 1;  % 防止除以0，导致NaN
Xtrain = (Xtrain - Xtrain_mean) ./ Xtrain_std;

global SVMModel
SVMModel = fitcecoc(Xtrain, labels);
% SVMModel = fitcecoc(Xtrain, labels, 'OptimizeHyperparameters', 'auto', ...
%     'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
%     'expected-improvement-plus'));
end


function Train_data(ampDir, Fs_target, band, leadIdx, winSec, db)
% 测试阶段 (使用提取的特征识别用户)  Testing phase (identify users using extracted features)

heaFiles = dir(fullfile(ampDir,'*.hea'));
recNames = unique(erase({heaFiles.name}, '.hea'));
groups = containers.Map;

for f = recNames
    m = regexp(f{1},'^(.*?_\d+)_\d+$','tokens');
    if isempty(m), continue; end
    key = m{1}{1};
    if isKey(groups,key)
        tmp = groups(key); tmp{end+1} = f{1}; groups(key) = tmp;
    else
        groups(key) = {f{1}};
    end
end

keys = groups.keys;
Xtest = []; % 测试特征 test features。
trueLabels = []; % 测试时的真实标签 ground truth labels during testing
userID = 1;

global Xtrain_mean Xtrain_std SVMModel mima

for n = 1:numel(keys)
    recSet = groups(keys{n});
    if isempty(recSet)
        continue;
    end

    % 只取每个人最后一条记录 only take the last record for each person。
    [ecg, Fs0] = loadAmpECG_manual(ampDir, recSet{end}, leadIdx);
    if Fs0 ~= Fs_target
        ecg = resample(ecg, Fs_target, Fs0);
    end
    ecg = denoiseECG(ecg, Fs_target);

    % 提取最后20秒的特征 extract features from the last 20 seconds
    if length(ecg) > Fs_target*winSec
        ecg_seg = ecg(end - Fs_target*winSec + 1 : end);
    else
        ecg_seg = ecg(1:min(end, Fs_target*winSec));
    end

    feat = extractFeatures(ecg_seg, Fs_target);
    vec = feat.vec;
    if any(isnan(vec))
        fprintf('[WARN] Test sample for user %d has NaN, skipping.\n', userID);
        continue;
    end

    Xtest = [Xtest; feat.vec];
    trueLabels = [trueLabels; userID];

    userID = userID + 1;
end

% 标准化测试特征 standardize test features
Xtest = (Xtest - Xtrain_mean) ./ Xtrain_std;

% 预测prediction
[predictedLabels, scores] = predict(SVMModel, Xtest);

% 打印结果 print results。
for idx = 1:length(predictedLabels)
    fprintf('[%2d] Known user #%d -> Predicted user #%d   pwd=%s\n', ...
        idx, trueLabels(idx), predictedLabels(idx), mima(predictedLabels(idx),:));
end

end



%% ---------- 文件读取 file reading----------
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
wname = 'coif4';    % 小波类型wavelet type。
level = 7;          % 分解尺度decomposition scale
[C, L] = wavedec(sig, level, wname); % 多尺度分解 multi-scale decomposition

a7_start = 1;                  % A7在C中的起始位置 Starting position of A7 in C。
a7_end = L(1);                 % A7在C中的结束位置 Ending position of A7 in C。
C_zeroA7 = C;                  % 复制系数 copy coefficients
C_zeroA7(a7_start:a7_end) = 0; % 将A7部分置零Set the A7 component to zero 

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
