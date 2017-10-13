%% Demo: fMRI LDA classifier & RSA with ROI & searchlight
%
% This demo is based on CosmoMVPA demo_fmri_searchlight_lda.m.
% Extended by Maureen Ritchey, October 2017.
%
% #   For CoSMoMVPA's copyright information and license terms,   #
% #   see the COPYING file distributed with CoSMoMVPA.           #

%% Data information

%%% Base directory
data_path='/Users/Maureen/DriveBC/Research/CosmoDemo/';
save_path = [data_path '/mvpa_results'];
if ~exist(save_path,'dir'), mkdir(save_path); end

%%% SPM file to load in
spm_fn = [data_path '/singletrial_model/SPM.mat'];

%%% Masks for ROI analysis & whole-brain analysis
mask_fn=fullfile(data_path,'rVTC_mask.nii');
wbmask_fn=fullfile(data_path,'rwholebrain_mask.nii');

%% Load data into CosmoMVPA

%%% Loading ROI data into CosmoMVPA - use SPM betas
% Note that loading data through the SPM.mat file will automatically
% "chunk" by runs, which is what we want
fprintf('Loading data from ROI: %s\n',mask_fn);
ds_roi=cosmo_fmri_dataset([spm_fn ':beta'],'mask',mask_fn);

%%% Tidy up the dataset
% Remove constant features
ds_roi=cosmo_remove_useless_data(ds_roi);

% Print dataset
fprintf('Dataset input:\n');
cosmo_disp(ds_roi);
fprintf('Number of samples: %i\n',size(ds_roi.samples,1));
fprintf('Number of features (voxels): %i\n',size(ds_roi.samples,2));
fprintf('Number of chunks (runs): %i\n',length(unique(ds_roi.sa.chunks)));

%%% Define trial information
% We want to classify faces vs scenes, so we need a vector that tells cosmo
% which betas (from our SPM model) have faces and which have scenes.
% Luckily that information is stored in our beta regressor labels, which
% are now contained in the dataset.

% Get face & scene information
scenes = strfind(ds_roi.sa.labels,'scene');
sceneidx = find(not(cellfun('isempty', scenes)));
faces = strfind(ds_roi.sa.labels,'face');
faceidx = find(not(cellfun('isempty', faces)));

sceneface = zeros(size(ds_roi.samples,1),1);
sceneface(sceneidx) = 1;
sceneface(faceidx) = 2;

% Tell Cosmo that you want the sceneface vector to be the targets (targets
% are whatever you want to classify)
ds_roi.sa.targets = sceneface;
fprintf('Number of possible targets: %i\n',length(unique(ds_roi.sa.targets)));

% Get fame information too; we'll use it later to split the analysis
fame = strfind(ds_roi.sa.labels,'_fame');
fameidx = find(not(cellfun('isempty', fame)));
nonfame = strfind(ds_roi.sa.labels,'nonfame');
nonfameidx = find(not(cellfun('isempty', nonfame)));

famenonfame = zeros(size(ds_roi.samples,1),1);
famenonfame(fameidx) = 1;
famenonfame(nonfameidx) = 2;

%%% Whole-brain data
% same procedure as above, but now we can directly assign the targets since
% we've already identified them
fprintf('Loading whole-brain data: %s\n',wbmask_fn);
ds_wb=cosmo_fmri_dataset([spm_fn ':beta'],'mask',wbmask_fn,'targets',sceneface);

% Remove constant features
ds_wb=cosmo_remove_useless_data(ds_wb);

% Print dataset
fprintf('Dataset input:\n');
cosmo_disp(ds_wb);


%% LDA classifier ROI analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis calculates classification accuracy using all voxels within
% the ROI. The classifier uses an n-fold partitioning scheme
% (cross-validation) and a Linear Discriminant Analysis (LDA) classifier.

% Define which classifier to use, using a function handle.
% Alternatives are @cosmo_classify_{svm,matlabsvm,libsvm,nn,naive_bayes}
classifier = @cosmo_classify_lda;

% Set partition scheme. odd_even is fast; for publication-quality analysis
% nfold_partitioner (cross-validation) is recommended.
% Alternatives are:
% - cosmo_nfold_partitioner    (take-one-chunk-out crossvalidation)
% - cosmo_nchoosek_partitioner (take-K-chunks-out  "             ").
% We will also make sure the partitions are *balanced* (targets are evenly
% distributed).
partitions = cosmo_nfold_partitioner(ds_roi);
partitions=cosmo_balance_partitions(partitions, ds_roi);

fprintf('There are %d partitions\n', numel(partitions.train_indices));
fprintf('# train samples:%s\n', sprintf(' %d', cellfun(@numel, ...
    partitions.train_indices)));
fprintf('# test samples:%s\n', sprintf(' %d', cellfun(@numel, ...
    partitions.test_indices)));

% Set any other options - see help cosmo_crossvalidate
opt = struct();
opt.normalization = 'zscore';
opt.max_feature_count = 7000; % just for the sake of the demo

% Run classification
[predictions,accuracy] = cosmo_crossvalidate(ds_roi,classifier,partitions,opt);

% Report results
fprintf('Accuracy: %0.2f\n',accuracy);


%% LDA classifier searchlight analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis identifies brain regions that discriminate the categories
% using a whole-brain searchlight procedure. The classifier uses an n-fold
% partitioning scheme (cross-validation) and a Linear Discriminant Analysis
% (LDA) classifier.

% Use the cosmo_cross_validation_measure and set its parameters
% (classifier and partitions) in a measure_args struct.
measure = @cosmo_crossvalidation_measure;
measure_args = struct();

% Define which classifier to use, using a function handle.
% Alternatives are @cosmo_classify_{svm,matlabsvm,libsvm,nn,naive_bayes}
measure_args.classifier = @cosmo_classify_lda;

% Set partition scheme. odd_even is fast; for publication-quality analysis
% nfold_partitioner is recommended.
% Alternatives are:
% - cosmo_nfold_partitioner    (take-one-chunk-out crossvalidation)
% - cosmo_nchoosek_partitioner (take-K-chunks-out  "             ").
partitions = cosmo_nfold_partitioner(ds_wb);
partitions=cosmo_balance_partitions(partitions, ds_wb);

fprintf('There are %d partitions\n', numel(partitions.train_indices));
fprintf('# train samples:%s\n', sprintf(' %d', cellfun(@numel, ...
    partitions.train_indices)));
fprintf('# test samples:%s\n', sprintf(' %d', cellfun(@numel, ...
    partitions.test_indices)));
measure_args.partitions = partitions;

% Define a neighborhood with approximately 100 voxels in each searchlight.
nvoxels_per_searchlight=100;
nbrhood=cosmo_spherical_neighborhood(ds_wb,...
    'count',nvoxels_per_searchlight);

% Run the searchlight
lda_results = cosmo_searchlight(ds_wb,nbrhood,measure,measure_args);

% print output dataset
fprintf('Dataset output:\n');
cosmo_disp(lda_results);

% Plot the output
cosmo_plot_slices(lda_results);

% Define output location
output_fn=fullfile(save_path,'lda_searchlight_faces_vs_scenes.nii');

% Store results to disc
cosmo_map2fmri(lda_results, output_fn);


%% LDA classifier ROI analysis - split by condition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis calculates classification accuracy using all voxels within
% the ROI. The classifier uses an n-fold partitioning scheme
% (cross-validation) and a Linear Discriminant Analysis (LDA) classifier.
% Analysis is limited to one condition at a time.

ncond = max(famenonfame);

% Loop over conditions
for cond = 1:ncond
    
    % Limit dataset to only trials matching current condition
    condidx = find(famenonfame==cond);
    ds_cond = cosmo_slice(ds_roi,condidx,1);
    
    % Define which classifier to use, using a function handle.
    classifier = @cosmo_classify_lda;
    
    % Set partition scheme
    partitions = cosmo_nfold_partitioner(ds_cond);
    partitions=cosmo_balance_partitions(partitions, ds_cond);
    
    % Set any other options - see help cosmo_crossvalidate
    opt = struct();
    opt.normalization = 'zscore';
    opt.max_feature_count = 7000; % just for the sake of the demo
    
    % Run classification
    [predictions,accuracy] = cosmo_crossvalidate(ds_cond,classifier,partitions,opt);
    
    % Report results
    fprintf('Condition %i ',cond);
    fprintf('Accuracy: %0.2f\n',accuracy);
    
end

%% Representational similarity ROI analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis compares the actual pattern dissimilarity matrix to a target
% dissimilarity matrix that assumes faces are more similar to other faces than
% they are to scenes (and vice versa). Computes actual pattern dissimilarity
% across the entire ROI.

%%% First define target DSM
% This matrix should be samples x samples. Note that it will default to
% using everything, so if you want to exclude within-run connections, you
% must set them to NaN.
% Here I'm brute-forcing the matrix by iterating through everything but
% there are more elegant ways to do this!

target_dsm = zeros(size(ds_roi.samples,1),size(ds_roi.samples,1));
for i=1:size(target_dsm,1)
    for j=1:size(target_dsm,2);
        
        if ds_roi.sa.chunks(i)==ds_roi.sa.chunks(j)         % drop within-run connections
            target_dsm(i,j) = NaN;
        elseif ds_roi.sa.targets(i)==ds_roi.sa.targets(j)   % same condition, zero distance (max similarity)
            target_dsm(i,j) = 0;
        else                                                % everything else is rated as different
            target_dsm(i,j) = 1;
        end
        
    end
end

% Show the target DSM
figure;
imagesc(target_dsm,[-1 1]);

%%% Run the RSA
measure_args=struct();
measure_args.target_dsm=target_dsm;
measure_args.type = 'Spearman'; % for corr between target & actual
measure_args.center_data = 1; % subtract voxel means

rsa_results = cosmo_target_dsm_corr_measure(ds_roi,measure_args);

fprintf('Correlation between target DSM & actual DSM: %0.2f\n',rsa_results.samples);

%% Representational similarity analysis - Searchlight
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis compares the observed pattern dissimilarity matrix to a target
% dissimilarity matrix that assumes faces are more similar to other faces than
% they are to scenes (and vice versa). Computes the actual pattern
% dissimilarity within searchlight neighborhoods across the whole brain.

%%% First define the target DSM
% I'll just use the one defined in the previous example.

%%% Define a neighborhood with approximately 100 voxels in each searchlight.
nvoxels_per_searchlight=100;
nbrhood=cosmo_spherical_neighborhood(ds_wb,...
    'count',nvoxels_per_searchlight);

%%% Run the RSA
measure = @cosmo_target_dsm_corr_measure;
measure_args=struct();
measure_args.target_dsm = target_dsm;
measure_args.type = 'Spearman'; % for corr between target & actual
measure_args.center_data = 1; % subtract voxel means

[ds_rsa] = cosmo_searchlight(ds_wb,nbrhood,measure,measure_args);

% print output dataset
fprintf('Dataset output:\n');
cosmo_disp(ds_rsa);

% Plot the output
cosmo_plot_slices(ds_rsa);

% Define output location
output_fn=fullfile(save_path,'rsa_searchlight_faces_vs_scenes.nii');

% Store results to disc
cosmo_map2fmri(ds_rsa, output_fn);


%% Representational similarity ROI analysis - Item-specific
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis compares split-half pattern similarity to a target
% similarity matrix that assumes individual scenes are more similar to a
% repetition of the same scene than to different scenes.

%%% Get ID information
% I need to get the stimulus ID information from another .mat file - it
% is not contained in the SPM.mat files.
id_fn = [data_path 'singletrial_model/all_trialinfo.mat'];
load(id_fn);
trial_ids = cell2mat(trialinfo(2:end,8));
trial_ids(trial_ids==999) = []; % I had labeled covariates (rp) as 999, remove those

% Update the dataset with new target & chunk information
ds_roi_id = ds_roi;
ds_roi_id.sa.targets = trial_ids;
ds_roi_id.sa.chunks(ds_roi.sa.chunks<=3) = 1; % first repetition
ds_roi_id.sa.chunks(ds_roi.sa.chunks>3) = 2; % second repetition

% Limit to scenes
sceneidx = find(sceneface==1);
ds_roi_id = cosmo_slice(ds_roi_id,sceneidx);
cosmo_disp(ds_roi_id);

%%% Define target DSM
% For this type of analysis (cosmo_correlation_measure), the first half is
% going to get matched up with the second half - so the target DSM is the n
% targets x n targets. Another different thing about this method is that all
% values must sum to zero.
n_targets = length(unique(ds_roi_id.sa.targets));
target_dsm = zeros(n_targets,n_targets);
target_dsm(eye(n_targets)==1) = 1/n_targets;
target_dsm(eye(n_targets)==0) = -1/(n_targets*(n_targets-1));

% Show the target DSM
figure;
imagesc(target_dsm);

%%% Run the RSA
% note that if template is not defined, it actually defaults to the
% diagonal - but here I'm using it just to show how it's done
measure_args=struct();
measure_args.template=target_dsm;
rsa_id_results = cosmo_correlation_measure(ds_roi_id,measure_args);
fprintf('Correlation between target DSM & actual DSM: %0.2f\n',rsa_id_results.samples);

% Let's do it again and get the raw matrix
measure_args=struct();
measure_args.template=target_dsm;
measure_args.output = 'raw';
rsa_id_results = cosmo_correlation_measure(ds_roi_id,measure_args);
rsa_mat = cosmo_unflatten(rsa_id_results,1);

figure;
imagesc(rsa_mat);

