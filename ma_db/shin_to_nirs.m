
%% specify subject
sbjid = "01";

%% Paths
paths.root = "C:\Users\avonl\OneDrive\Desktop\data\";
paths.sbjdir = "subject "+ sbjid + "\";
paths.target = paths.root + paths.sbjdir + "snirf\"
paths.data = "cnt.mat"
paths.montage = "mnt.mat"
paths.stim = "mrk.mat"

addpath(genpath(paths.root))

%% load data for subject
load(paths.root+paths.sbjdir+paths.data);
load(paths.root+paths.sbjdir+paths.montage);
load(paths.root+paths.sbjdir+paths.stim);

%% Create fields in nirs file, create one file per recording
% documented here https://www.nmr.mgh.harvard.edu/martinos/software/homer/HOMER2_UsersGuide_121129.pdf

for i=1:numel(cnt)
% sample rate (scalar)
    tmp.fs = cnt{i}.fs;
% t (time, T x 1)
    tmp.t = 0:1:size(cnt{i}.x,1)-1;
    tmp.t = tmp.t/cnt{i}.fs;
    tmp.t=tmp.t'
% d (data, T x CH)
    tmp.d = cnt{i}.x;
% SD (Source Detector geometry Structure
    % Wavelenghts (n_lambda x 1)
    tmp.SD.Lambda = cnt{i}.wavelengths;
     % Number of Sources (scalar)
    tmp.SD.nSrcs = cnt{i}.nSources;
    % Number of Detectors (scalar)
    tmp.SD.nDets = cnt{i}.nDetectors;
    % create measurement List and deal with mess in Shin's data (NaNs
    % require reindexing)
    ML = mnt.sd;                  % original measurements, nChan×4
    Src = mnt.source.pos_3d';          % nSrcs×3
    % iterate from last row down to 1
    for r = size(Src,1):-1:1
        if any( isnan(Src(r,:)) )
            % 1) remove that source row
            Src(r,:) = [];
            % 2) decrement all ML(:,1) entries that pointed after r
            mask = ML(:,1) > r;
            ML(mask,1) = ML(mask,1) - 1;
        end
    end
    % stick cleaned geometry back
    tmp.SD.SrcPos = Src;
    % do the same for detectors
    MLd = ML;                     % work off the same ML, or reload if separate
    Det = mnt.detector.pos_3d';
    for r = size(Det,1):-1:1
        if any( isnan(Det(r,:)) )
            Det(r,:) = [];
            mask = MLd(:,2) > r;
            MLd(mask,2) = MLd(mask,2) - 1;
        end
    end
    tmp.SD.DetPos = Det;
    % to drop any channels that referred to a now-deleted source or detector:
    valid = (ML(:,1)>0 & MLd(:,2)>0);
    ML = ML(valid,:);
    % now add wavelength columns as before:
    ML(:,3) = 1; 
    ML(:,4) = 1;
    ML2 = ML; ML2(:,4)=2;
    tmp.SD.MeasList = [ML; ML2];

% stimulus onsets s (n_timepoints x n_conditions), 1 at stimulus onset at
% time point t, 0 otherwise
    tmp.s = zeros(numel(tmp.t), size(mrk{i}.y, 1)); % initialize with zeros
    % extract time points for each condition and get to same time base
    stims = round(mrk{i}.time/1000,1);
    % for each condition 
    for cc=1:size(mrk{i}.y,1)
        c_idx = find(mrk{i}.y(cc,:))
        sidx = ismember(tmp.t, stims(c_idx));     % logical mask of length T
        tmp.s(sidx,cc) = 1;
    end
% ml (list of source-detector channels, duplicate of SD.MeasList)
    tmp.ml = tmp.SD.MeasList;
% store condition names
tmp.CondNames = mrk{i}.className;

% save nirs file
    if ~exist(paths.target,'dir')
        mkdir(paths.target)
    end
    savepath = fullfile( paths.target, sprintf('sbj_%s_run0%d.nirs', sbjid, i) );
    save(savepath, '-struct', 'tmp')
end
