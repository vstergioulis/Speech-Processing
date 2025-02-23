clear; clc;


directory_root = 'AudioMNIST_Speech';
subdirs = {'Train', 'Test'};

% Parameters
frame_len = 20; % ms
overlap = 5; %ms
NUM_FEAT = 20;
NUM_CLASSES = 5; % 0,1,2,3,4
NUM_MIXTURES = 10; % we will use 10 mixtures for every class


%% Feature EXT
for d = 1:length(subdirs)
    rootDir = fullfile(directory_root, subdirs{d});
    subDirs = dir(fullfile(rootDir, '*')); % Get subdirectories

    % Filter out non-directory entries (., .., etc.)
    subDirs = subDirs([subDirs.isdir]);
    subDirs = subDirs(~ismember({subDirs.name}, {'.', '..'}));

    % Loop through each subdirectory (0,1,2,...)
    for i = 1:length(subDirs)
        subDirPath = fullfile(rootDir, subDirs(i).name);
        wavFiles = dir(fullfile(subDirPath, '*.wav')); % Get all wav files

        % Loop through each wav file
        for j = 1:length(wavFiles)
            filePath = fullfile(subDirPath, wavFiles(j).name); % Construct full path

            if contains(filePath, 'Train')
                split = 'Train';
            else
                split = 'Test';
            end

            extract_features(filePath,split,frame_len,overlap, NUM_FEAT);
        end
    end
end

disp('Features Extracted')
disp('~~~~~~~~~~~~~~~~~~~~~~~')
disp('TRAINING')
disp('~~~~~~~~~~~~~~~~~~~~~~~')
%% TRAINING FOR GMMS
train_dir = dir(fullfile('Mel_Features/train', '*mat'));
test_dir = dir(fullfile('Mel_Features/test', '*mat'));

% Get every class instance

% initialize lists that contain the correct samples
mfccs_0 = []; mfccs_1 = []; mfccs_2 = []; mfccs_3 = []; mfccs_4 = [];

gmm_models = cell(NUM_CLASSES,1);
num_frames = 50;
for k = 1:length(train_dir)
    filepath = fullfile(train_dir(k).folder, train_dir(k).name);
    filename = train_dir(k).name;

    % Get our label-class
    class = filename(1);
    feat = load(filepath);
    feat = feat.MFCCs;
    feat = feat.';
    switch class
        case '0'
            mfccs_0 = [mfccs_0; feat(1:num_frames,:)];
        case '1'
            mfccs_1 = [mfccs_1; feat(1:num_frames,:)];
        case '2'
            mfccs_2 = [mfccs_2; feat(1:num_frames,:)];
        case '3'
            mfccs_3 = [mfccs_3; feat(1:num_frames,:)];
        case '4'
            mfccs_4 = [mfccs_4; feat(1:num_frames,:)];
        otherwise
            disp('NO CLASS')
    end
end

%% TRAINING EVERY MODEL
% Correctly pass mfccs for our Gaussian Mixtures
% Input FxN, F: Num of features, N: Number of points
[~,gmm_models{1},llh_0] = mixGaussEm(mfccs_0.', NUM_MIXTURES);
[~,gmm_models{2},llh_1] = mixGaussEm(mfccs_1.', NUM_MIXTURES);
[~,gmm_models{3},llh_2] = mixGaussEm(mfccs_2.', NUM_MIXTURES);
[~,gmm_models{4},llh_3] = mixGaussEm(mfccs_3.', NUM_MIXTURES);
[~,gmm_models{5},llh_4] = mixGaussEm(mfccs_4.', NUM_MIXTURES);

% Our gmm_models{i}, contains the mean Value, Variance and weights for
% every Gaussian Mixture of every model, time to save these values

for i = 1:5
    model_par = cell2mat(gmm_models(i));
    fileout = ['GMM/', 'mean' ,num2str(i-1), '.mat'];
    mean = model_par.mu;
    save(fileout,'mean');

    fileout = ['GMM/', 'variance' ,num2str(i-1), '.mat'];
    sigma = model_par.Sigma;
    save(fileout,'sigma');

    fileout = ['GMM/', 'weights' ,num2str(i-1), '.mat'];
    weight = model_par.w;
    save(fileout,'weight');
end
disp('~~~~~~~~~~~~~~~~~~~~~~~')
disp('SAVING GMM PARAMETERS: MEAN, VARIANCE & WEIGHT')
disp('~~~~~~~~~~~~~~~~~~~~~~~')

%% Classification
correct_preds = 0;

disp('~~~~~~~~~~~~~~~~~~~~~~~')
disp('CLASSIFICATION')
disp('~~~~~~~~~~~~~~~~~~~~~~~')
for k = 1:length(test_dir)
    filepath = fullfile(test_dir(k).folder, test_dir(k).name);
    filename = test_dir(k).name;

    % Get our label-class
    real_class = filename(1);
    feat = load(filepath);
    feat = feat.MFCCs;
    mfcc = feat.';
    

    % Now that we created the test dataset, it is time to use MAP to determine
    % to which class we classify our data
    likelihoods = zeros(1,NUM_CLASSES);

    for i = 1:5
        % iterate through every class model and get the loglikelihoods
        model_par = cell2mat(gmm_models(i));
        mean = model_par.mu;
        sigma = model_par.Sigma;
        weight = model_par.w;

        llh__ = maxapost(mfcc, mean, sigma, weight);
        likelihoods(i) = sum(log(llh__ + 1e-6)); 
    end
    
    [~,predicted_class_index] = max(likelihoods);
    % here predicted class will contain the index of the highest
    % loglikelihoo, i.d. if the highest llh is in position 3, then
    % predicted_class_index = 3

    predicted_class = num2str(predicted_class_index - 1); % because we have classes 0,1,2,3,4

    if strcmp(real_class,predicted_class)
        correct_preds = correct_preds+1;
    end
    
end

accuracy = (correct_preds/length(test_dir))*100 ;

disp(['Total Accuracy for using ', num2str(NUM_FEAT), ' MFCCs,', num2str(num_frames), ' Frames and ', num2str(NUM_MIXTURES), ' mixtures for every class, is ', num2str(accuracy), '%'])