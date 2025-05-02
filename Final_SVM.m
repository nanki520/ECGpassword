% ======================================================================
%  ECG_Auth_System.m      rev-I  (2025-04-24)
% ======================================================================
clc; clear; close all;

%% =================== 全局参数 global parameters。===================
Fs_target = 500;            % 统一采样率 uniform sampling rate
winSec  = 10;             % 窗口长度 (秒)window length (seconds)。
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




%% ---------- AMP 初始化 pre-initialize AMP----------
function db = initUsersFromAMP(ampDir, Fs_target, band, leadIdx, winSec, db)
heaFiles = dir(fullfile(ampDir,'*.hea'));
recNames = unique( erase({heaFiles.name}, '.hea') );
groups   = containers.Map;

for f = recNames
    m = regexp(f{1},'^(.*?_\d+)_\d+$','tokens'); if isempty(m), continue; end
    key = m{1}{1};
    if isKey(groups,key)
        tmp = groups(key); tmp{end+1} = f{1}; groups(key) = tmp;
    else
        groups(key) = {f{1}};
    end
end

keys = groups.keys;

Xtrain = [];
lable  = [];
kk = 1;
kkk = 1;
Xall = [];
global mima
mima = [];

for n = 1:numel(keys)-1
    recSet = groups(keys{n});
    for r = recSet(1:(length(recSet)))

        [ecg,Fs0] = loadAmpECG_manual(ampDir,r{1},leadIdx);
        if Fs0~=Fs_target, ecg = resample(ecg,Fs_target,Fs0); end
        ecg = denoiseECG(ecg,Fs_target);
        feat = extractFeatures(ecg(1:Fs_target*winSec),Fs_target);
        feat.traindata = ecg(1:Fs_target*winSec);
        Xall = [Xall; feat.vec];
        if(length(ecg)>100000)
            for kkkkk= 1:Fs_target*winSec :(length(ecg)-Fs_target*winSec )
                lable(kk) = kkk;
                kk = kk+1;
                Xtrain = [Xtrain; ecg(kkkkk:(kkkkk+winSec*Fs_target -1))];
            end
        else
            lable(kk) = kkk;
            kk = kk+1;
            Xtrain = [Xtrain; feat.traindata];
        end
    end

    mima(kkk,:)   = generatePassword(mean(Xall,1));
    kkk = kkk+1;
end

% 标准化训练数据normalize training data
Xtrain = (Xtrain - mean(Xtrain,1)) ./ std(Xtrain,[],1);

global SVMModel
SVMModel = fitcecoc(Xtrain, lable);
% SVMModel = fitcecoc(Xtrain, lable, 'OptimizeHyperparameters', 'auto', ...
%     'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
%     'expected-improvement-plus'));
end


function  Train_data(ampDir, Fs_target, band, leadIdx, winSec, db)
heaFiles = dir(fullfile(ampDir,'*.hea'));
recNames = unique( erase({heaFiles.name}, '.hea') );
groups   = containers.Map;
for f = recNames
    m = regexp(f{1},'^(.*?_\d+)_\d+$','tokens'); if isempty(m), continue; end
    key = m{1}{1};
    if isKey(groups,key)
        tmp = groups(key); tmp{end+1} = f{1}; groups(key) = tmp;
    else
        groups(key) = {f{1}};
    end
end

global SVMModel Xtrain_mean Xtrain_std mima
keys = groups.keys;
Xtrain = [];
Xall = [];

for n = 1:numel(keys)
    recSet = groups(keys{n});
    for r = recSet(length(recSet))%只取最后一条记录。Only take the last record.
        [ecg,Fs0] = loadAmpECG_manual(ampDir,r{1},leadIdx);
        if Fs0~=Fs_target, 
        ecg = resample(ecg,Fs_target,Fs0); end
        ecg = denoiseECG(ecg,Fs_target);

        %feat = extractFeatures(ecg(1:Fs_target*winSec),Fs_target);
        %Xall = [Xall; feat.vec];
        %feat.traindata = ecg(1:Fs_target*winSec);

        if(length(ecg)>100000)
            for kkkkk= length(ecg)-Fs_target*winSec +1:length(ecg)Fs_target*winSec +1
                % lable(kk) = kkk;
                % kk = kk+1;
                Xtrain = [Xtrain; ecg(kkkkk:(kkkkk+Fs_target*winSec -1))];
            end
        else
            %Xtrain = [Xtrain; feat.traindata];
            Xtrain = [Xtrain; ecg(1:Fs_target*winSec)];
        end
    end

end

% 测试数据标准化Standardize test data
Xtrain_mean = mean(Xtrain,1);
Xtrain_std = std(Xtrain,[],1);
Xtrain = (Xtrain - Xtrain_mean) ./ Xtrain_std;

[predictedLabels,sc] = predict(SVMModel, Xtrain)
global mima
for kkkk = 1:length(predictedLabels)
    fprintf('[%2d] %-25s  Known user #%d  pwd=%s\n', ...
        kkkk, 1, predictedLabels(kkkk), mima(predictedLabels(kkkk),:));
end

end


%% ---------- 文件读取file reading ----------
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
[C, L] = wavedec(sig, level, wname); % 多尺度分解multi-scale decomposition

a7_start = 1;                  % A7在C中的起始位置 Starting position of A7 in C。
a7_end = L(1);                 % A7在C中的结束位置 Ending position of A7 in C。
C_zeroA7 = C;                  % 复制系数copy coefficients
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
