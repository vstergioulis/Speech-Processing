function [frame_llh] = maxapost(mfcc,mean,variance,weight)

% Initalize the Log-Likelihood for every frame as 0. 
frame_llh = zeros(size(mfcc,1),1); 

for k = 1:length(weight)
    frame_llh = frame_llh + weight(k)*mvnpdf(mfcc, mean(:,k).' , variance(:,:,k).');
end

% % Now that we have calculated the llh for every frame we need to sum
% % them
% loglh = sum(log(frame_llh + 1e-6)); 
% % we add 1, in case our frame_llh is close to 0 and avoid -inf as a possible log likelihood value 
end

