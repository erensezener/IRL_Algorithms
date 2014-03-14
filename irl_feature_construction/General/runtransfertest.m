% Run IRL transfer test with specified IRL result and example.
function test_result = runtransfertest(irl_result, algorithm, ...
    mdp_model, problem, params)

% test_result - structure that contains results of the test:
%   see evaluateirl.m
% irl_result - input IRL result
% mdp_model - string specifying MDP model to use for examples:
%   standardmdp - standard MDP model
% problem - string specifying example to test on:
%   gridworld
% mdp_params - string specifying parameters for example:
%   Gridworld:
%       seed (0) - initialization for random seed
%       n (32) - number of cells along each axis
%       b (4) - size of macro cells
%       determinism (1.0) - probability of correct transition
%       discount (0.9) - temporal discount factor to use
% test_params - general parameters for the test:
%   test_models - models to test on
%   test_metrics - metrics to use during testing

mdp_params       = params.mdp;
test_params      = params.test;
mdp_params.seed  = params.seed;
test_params.seed = params.seed;

% Set default test parameters.
test_params = setdefaulttransferparams(test_params);

% Construct MDP and features.
[mdp_data, r, feat_data, true_feats] = feval(strcat(problem, 'build'), mdp_params);

% Unroll IRL result.
irl_result = feval(strcat(algorithm, 'transfer'), irl_result, mdp_data, mdp_model, ...
    feat_data, true_feats, test_params.verbosity);

% Evaluate result.
test_result = evaluateirl(irl_result, r, [], mdp_data, mdp_params, problem, ...
    test_params.test_models, test_params.test_metrics, ...
    feat_data, true_feats);
for i = 1:length(test_result)
    test_result(i).algorithm = algorithm;
end
