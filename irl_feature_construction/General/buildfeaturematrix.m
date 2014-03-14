function F = buildfeaturematrix(mdp_data, feature_data, true_features, params)

nS = size(mdp_data.sa_p, 1);

if isstruct(params)     
    % test case
    if params.all_features,
        F = feature_data.splittable;    % Add all base features
        F = horzcat(F, ones(nS, 1));    % Add dummy feature
    elseif algorithm_params.true_features,
        F = true_features;
    else
        F = eye(nS);
    end
else
    % trasfer case
    w = params;
    
    if length(w) == size(true_features, 2)
        F = true_features;
    elseif length(w) == size(feature_data.splittable, 2) + 1
        F = feature_data.splittable;
        F = horzcat(F, ones(nS, 1));
    elseif length(w) == size(feature_data.splittable, 2)*2 + 1
        F = horzcat(feature_data.splittable, ~feature_data.splittable);
        F = horzcat(F, ones(nS, 1));
    else
        F = eye(nS);
    end
end

end