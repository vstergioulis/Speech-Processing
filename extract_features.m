function [] = extract_features(filename, split,frame_length, overlap, num_of_features)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    L = 22;                 % cepstral sine lifter parameter
    LF = 300;               % lower frequency limit (Hz)
    HF = 3700;              % upper frequency limit (Hz)
    
    [ ~ ,name, ~] = fileparts(filename);
    [signal, Fs] = audioread(filename);
    [MFCCs,FBEs,frames ] = mfcc(signal, Fs, frame_length, overlap, alpha, @hamming, [LF HF], M, num_of_features+1, L );
    MFCCs = MFCCs(2:end,:);  % discard the first column or MFCC_0, due to its high correlation
                             % with energy
    
    if strcmp(split,'Test')
        filename_output = ['Mel_Features/', split,'/',name, '.mat'];
    else
        filename_output = ['Mel_Features/', split,'/',name, '.mat'];
    end

    save(filename_output,'MFCCs')

end

