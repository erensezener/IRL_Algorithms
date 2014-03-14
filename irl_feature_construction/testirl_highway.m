%% Script for testing IRL algorithms

clear all;
addpaths;
% matlabpool open 10;

% algorithm  = 'npbfirl';
% algorithm  = 'gpbirl';
% algorithm  = 'gpirl';
algorithm  = 'firl';
% algorithm  = 'maxent'; %'mmp'; 
% algorithm = 'birlga'; %'birlmcmc'; %

ntrials      = 10;
ntransfers   = 10;
nsamples     = 35; %[10, 20, 30, 40, (50)];
samplelength = 200;
sampletype   = 'optimal'; % optimal, epsgreedy

mdp_model  = 'standardmdp';
problem    = 'highway';

%% Prepare parameters
alg_params = struct(...
    'eta',           10.0, ...  % confidence parameter of choosing optimal actions (5.0)
    'normal_prior',  1, ...    % use zero mean normal prior for regularization
    'sigma',         1.0, ...  % standard deviation of prior
    'alpha',         1.0, ...  % concentration parameter for IBP
    'beta',          1.0, ...  % concentration parameter for IBP
    'a',             1.0, ...  % parameter of sparse IBP
    'b',             9.0, ...  % parameter of sparse IBP
    'lambda',        1.0, ...  % parameter of proposal distribution for updating weight
    'return_mean',   1, ...    % return mean estimate or MAP estimate
    'max_iters',     400, ...  % maximum # of iterations of MCMC
    'seed',          0, ...
    'all_features',  true, ...
    'true_features', false);

mdp_params  = struct( ...
    'length',      4*64, ... %36, ... %64, ...
    'lanes',       3, ...
    'speeds',      4, ...
    'num_cars',    4*[6 34], ... %[4 14], ...%[34 6], ...
    'policy_type', 'outlaw', ... %'lawful', ... %
    'determinism', 0.7, ... %1.0, ...
    'continuous',  false);

test_params = struct(...
    'type',                    sampletype, ...
    'epsilon',                 0.2, ...
    'training_samples',        nsamples, ...
    'training_sample_lengths', samplelength, ...
    'verbosity',               1);

param_set = cell(ntrials, 1);
for iter = 1:ntrials
    param_set{iter}.alg  = alg_params;
    param_set{iter}.mdp  = mdp_params;
    param_set{iter}.test = test_params;
end

%% Run IRL algorithm
test_results     = cell(ntrials, 1);
transfer_results = cell(ntrials, ntransfers);
parfor iter = 1:ntrials
% for iter = 1:ntrials
    fprintf('## %d-th trial | %s - %s %s (%d %d %d)##\n', ...
        iter, algorithm, param_set{iter}.test.type, problem, ...
        param_set{iter}.mdp.length, ...
        param_set{iter}.mdp.num_cars(1), param_set{iter}.mdp.num_cars(2));

    param_set{iter}.seed = iter;    
    test_results{iter}   = runtest(algorithm, mdp_model, problem, param_set{iter});
    printresult(test_results{iter});
    
    irl_result = test_results{iter}.irl_result;
    for j = 1:ntransfers
        param_set{iter}.seed = iter*1000 + j;
        transfer_results{iter, j} = runtransfertest(irl_result, ...
            algorithm, mdp_model, problem, param_set{iter});
        printresult(transfer_results{iter, j});
    end    
    fprintf('## %d-th trial is finished ## \n\n', iter);
end

%% Print the statistics of the results
if isfield(test_results{1}.irl_result, 'map') ...
        && isfield(test_results{1}.irl_result, 'mean')
    [mapR1, mapR2]   = printresultstat(test_results, transfer_results, 'map', []);
    [meanR1, meanR2] = printresultstat(test_results, transfer_results, 'mean', []);
else
    [result1, result2] = printresultstat(test_results, transfer_results, [], []);
end

%% Plot the results of MCMC algorithm
if ~isempty(strfind(algorithm, 'mcmc')) || ~isempty(strfind(algorithm, 'npb'))
    fig = feval(strcat(algorithm, 'visualize'), test_results);
else
    fig = [];
end

%% Save results
outfname = [];
outpath  = sprintf('ExpResults3_%s/%s_%s', ...
    problem, test_params.type, datestr(now, 'yymmdd'));
if ~isdir(outpath)
    fprintf('Mkdir %s !!!\n\n', outpath);
    mkdir(outpath);
end
outfname = sprintf('traj%dx%d_%s', nsamples, samplelength, algorithm);
save(sprintf('%s/%s_test.mat', outpath, outfname), 'test_results', '-v7.3');
save(sprintf('%s/%s_transfer.mat', outpath, outfname), 'transfer_results', '-v7.3');
if ~isempty(fig)
    saveas(gcf, sprintf('%s/%s.fig', outpath, outfname));
    close(fig);
end

matlabpool close;
rmpaths;
