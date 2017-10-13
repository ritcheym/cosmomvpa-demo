%% Demo: fMRI searchlights LDA classifier
%
% This demo is based on CosmoMVPA demo_fmri_searchlight_lda.m.
%
% #   For CoSMoMVPA's copyright information and license terms,   #
% #   see the COPYING file distributed with CoSMoMVPA.           #

%% Data information


%% LDA classifier searchlight analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis identified brain regions where the categories can be
% distinguished using an odd-even partitioning scheme and a Linear
% Discriminant Analysis (LDA) classifier.

data_path=digit_study_path;
data_fn=fullfile(data_path,'glm_T_stats_perblock+orig.HEAD');
mask_fn=fullfile(data_path,'brain_mask+orig.HEAD');

targets=repmat(1:2,1,16)';    % class labels: 1 2 1 2 1 2 1 2 1 2 ... 1 2
chunks=floor(((1:32)-1)/8)+1; % run labels:   1 1 1 1 1 1 1 1 2 2 ... 4 4

ds_per_run = cosmo_fmri_dataset(data_fn, 'mask', mask_fn,...
                                'targets',targets,'chunks',chunks);

% print dataset
fprintf('Dataset input:\n');
cosmo_disp(ds_per_run);


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
measure_args.partitions = cosmo_oddeven_partitioner(ds_per_run);

% print measure and arguments
fprintf('Searchlight measure:\n');
cosmo_disp(measure);
fprintf('Searchlight measure arguments:\n');
cosmo_disp(measure_args);

% Define a neighborhood with approximately 100 voxels in each searchlight.
nvoxels_per_searchlight=100;
nbrhood=cosmo_spherical_neighborhood(ds_per_run,...
                        'count',nvoxels_per_searchlight);


% Run the searchlight
lda_results = cosmo_searchlight(ds_per_run,nbrhood,measure,measure_args);

% print output dataset
fprintf('Dataset output:\n');
cosmo_disp(lda_results);

% Plot the output
cosmo_plot_slices(lda_results);

% Define output location
output_fn=fullfile(output_path,'lda_searchlight+orig');

% Store results to disc
cosmo_map2fmri(lda_results, output_fn);

% Show citation information
cosmo_check_external('-cite');
