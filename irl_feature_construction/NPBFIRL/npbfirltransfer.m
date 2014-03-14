% Transfer learned reward function to a new state space.
function irl_result = npbfirltransfer(prev_result, mdp_data, mdp_model,...
    feature_data, true_features, verbosity)

% Get MAP estimate
map = prev_result.map;

% Get all samples
hst = prev_result.hst;

% Generate base features
if size(map.Z, 1) == size(true_features, 2)
    F = true_features;
elseif size(map.Z, 1) == size(feature_data.splittable, 2) + 1
    F = feature_data.splittable;
    F = horzcat(F,ones(mdp_data.states,1));
else
    F = eye(mdp_data.states);
end

% Generate sparse state-transition matrix and base features
mdp_data.TG = gendiscountedtransmtrx(mdp_data);

% # of samples to be discarded
hstlength = length(hst.logPost);
nsamples  = ceil(hstlength*0.5);

% Compute mean reward
mean.r = zeros(mdp_data.states, 1);
for i = (nsamples + 1):hstlength
    tmpr  = npbfirlgenfeatmtrx(hst.X{i}, F)*hst.X{i}.w;
    mean.r = mean.r + tmpr;
end
mean.r    = repmat(mean.r./(hstlength - nsamples), 1, mdp_data.actions);
mean.soln = npbfirlsolvemdp(mean.r, mdp_data, []);

% Compute MAP reward
map.r    = repmat(npbfirlgenfeatmtrx(map, F)*map.w, 1, mdp_data.actions);
map.soln = npbfirlsolvemdp(map.r, mdp_data, []);

% Build IRL result
irl_result = struct('time', 0, 'score', 0, 'map', map, 'mean', mean);

end