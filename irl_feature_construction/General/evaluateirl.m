% Evaluate result returned by IRL algorithm on given MDP.
% Modified by Jaedeug Choi
function test_result = evaluateirl(irl_result, true_r, example_samples, ...
    mdp_data, mdp_params, problem, test_models, test_metrics,...
    feature_data, true_feature_map)

% test_result: data structure encapsulating generic test results:
%   metric_scores   - array of results from each metric, for each MDP type
%   irl_result      - copy of result from IRL algorithm
%   true_r          - true reward function
%   example_samples - copy of example samples
%   mdp_data        - copy of MDP data
%   mdp_params      - copy of MDP parameters
%   mdp_solution    - copy of true MDP solution
%   feature_data    - copy of feature data
%   problem             - type of MDP used

if isfield(irl_result, 'map') && isfield(irl_result, 'mean')
    map_result   = irl_result;
    map_result.r = map_result.map.r;
    map_result.p = map_result.map.soln.p;
    map_result.v = map_result.map.soln.v;
    map_result.q = map_result.map.soln.q;
    map_scores   = evaluate(map_result, true_r, example_samples, ...
        mdp_data, mdp_params, test_models, test_metrics,...
        feature_data, true_feature_map);
    
    mean_result   = irl_result;
    mean_result.r = mean_result.mean.r;
    mean_result.p = mean_result.mean.soln.p;
    mean_result.v = mean_result.mean.soln.v;
    mean_result.q = mean_result.mean.soln.q;
    mean_scores   = evaluate(mean_result, true_r, example_samples, ...
        mdp_data, mdp_params, test_models, test_metrics,...
        feature_data, true_feature_map);
    
    metric_scores = struct( ...
        'map',  {map_scores}, ...
        'mean', {mean_scores});
else
    metric_scores   = evaluate(irl_result, true_r, example_samples, ...
        mdp_data, mdp_params, test_models, test_metrics,...
        feature_data, true_feature_map);
end

% Build results structure.
test_result = struct(...
    'metric_scores',   {metric_scores}, ...
    'irl_result',      irl_result, ...
    'true_r',          true_r, ...
    'example_samples', {example_samples}, ...
    'test_models',     {test_models}, ...
    'test_metrics',    {test_metrics}, ...
    'mdp_data',        mdp_data, ...
    'mdp_params',      feval(strcat(problem, 'defaultparams'), mdp_params), ...
    'feature_data',    feature_data, ...
    'problem',         problem);

end


% Evaluate result returned by IRL algorithm on given MDP.
function metric_scores = evaluate(irl_result, true_r, example_samples, ...
    mdp_data, mdp_params, test_models, test_metrics,...
    feature_data, true_feature_map)

metric_scores = cell(length(test_models), length(test_metrics));
for m = 1:length(test_models)
    % Evaluate for each test model.
    mdp_solve = str2func(strcat(test_models{m}, 'solve'));
    irl_soln  = mdp_solve(mdp_data, irl_result.r);
    mdp_soln  = mdp_solve(mdp_data, true_r);
    irl_r     = irl_result.r;
    
    % Evaluate each metric.
    for k = 1:length(test_metrics),
        cur_metric          = test_metrics{k};
        metric_scores{m, k} = feval(strcat(cur_metric, 'score'), ...
            mdp_soln, true_r, irl_soln, irl_r, feature_data, ...
            true_feature_map, mdp_data, mdp_params, test_models{m});
    end;
end;

end