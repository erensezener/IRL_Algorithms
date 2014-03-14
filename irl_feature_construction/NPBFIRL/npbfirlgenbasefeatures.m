% Generate base features
% Written by Jaedeug Choi
function F = npbfirlgenbasefeatures(feature_data, true_features, alg_params)

if alg_params.all_features
    F = feature_data.splittable;
    N = size(F, 1);
    F = horzcat(feature_data.splittable, ones(N, 1));  % Add dummy features
elseif alg_params.true_features
    F = true_features;
else
    F = eye(nS);
end

end