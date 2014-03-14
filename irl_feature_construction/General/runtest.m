% Run IRL test with specified algorithm and example.
function test_result = runtest(algorithm, mdp_model, problem, params)

% test_result: structure that contains results of the test: see evaluateirl.m
% algorithm:   string specifying the IRL algorithm to use
% alg_params:  parameters of the specified algorithm
% mdp_model:   string specifying MDP model to use for examples:
%   standardmdp - standard MDP model
% problem:         string specifying example to test on
% mdp_params:  string specifying parameters for example
% test_params: general parameters for the test:
%   test_models                   - models to test on
%   test_metrics                  - metrics to use during testing
%   training_samples (32)         - # of example trajectories to query
%   training_sample_lengths (100) - length of each sample trajectory
%   true_features ([])            - alternative set of true features

alg_params       = params.alg;
mdp_params       = params.mdp;
test_params      = params.test;
alg_params.seed  = params.seed;
mdp_params.seed  = params.seed;
test_params.seed = params.seed;

% Set default test parameters.
test_params = setdefaulttestparams(test_params);

% Construct MDP and features.
[mdp_data, r, feat_data, true_feats] = feval(strcat(problem, 'build'), mdp_params);
mdp_data.r = r;

% Solve example.
mdp_soln = feval(strcat(mdp_model, 'solve'), mdp_data, r);

% Sample example trajectories.
example_samples = sampleexamples(mdp_model, mdp_data, mdp_soln, test_params);

% Run IRL algorithm.
irl_result = feval(strcat(algorithm, 'run'), alg_params, mdp_data, mdp_model, ...
    feat_data, example_samples, true_feats, test_params.verbosity);

% Evaluate result.
test_result = evaluateirl(irl_result, r, example_samples, mdp_data, mdp_params, ...
    problem, test_params.test_models, test_params.test_metrics, ...
    feat_data, true_feats);

test_result.algorithm     = algorithm;
test_result.true_features = true_feats;
