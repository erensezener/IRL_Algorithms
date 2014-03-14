%% Script for testing IRL algorithms

clear all;
addpaths;

algorithm  = 'npbfirl';
% algorithm  = 'gpirl';
% algorithm  = 'firl';
% algorithm  = 'mmp'; %'maxent'; %
% algorithm = 'birlga';

ntrials      = 4;
ntransfers   = 0;
nsamples     = 100; %[10, 20, 30, 40, (50)];
samplelength = 200;
sampletype   = 'optimal'; % optimal, epsgreedy

mdp_model  = 'standardmdp';
problem    = 'gridworld'; %'objectworld';
ngrids     = 16; %32
ncolors    = 2; %[2, (4), 6, 8, 10];
blocksize  = 2;

%% Prepare parameters
alg_params = struct(...
    'eta',           5.0, ...  % confidence parameter of choosing optimal actions (5.0)
    'normal_prior',  1, ...    % use zero mean normal prior for regularization
    'sigma',         1.0, ...  % standard deviation of prior
    'alpha',         1.0, ...  % concentration parameter for IBP
    'beta',          1.0, ...  % concentration parameter for IBP
    'a',             1.0, ...  % parameter of sparse IBP
    'b',             9.0, ...  % parameter of sparse IBP
    'lambda',        1.0, ...  % parameter of proposal distribution for updating weight
    'return_mean',   1, ...    % return mean estimate or MAP estimate
    'max_iters',     300, ...  % maximum # of iterations of MCMC
    'seed',          0, ...
    'all_features',  1, ...
    'true_features', 0);

mdp_params = struct(...
    'n',           ngrids, ...
    'b',           blocksize, ...
    'c1',          ncolors, ...
    'seed',        0, ...
    'determinism', 0.7, ...
    'continuous',  0);

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
for iter = 1:ntrials
    fprintf('## %d-th trial | %s - %s grids:%d colors:%d samples:%d ##\n', ...
        iter, algorithm, param_set{iter}.test.type, ngrids, ncolors, nsamples);

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
outpath  = sprintf('ExpResults_%s/n%d_%s_%s', ...
    problem, ngrids, test_params.type, datestr(now, 'yymmdd'));
if ~isdir(outpath)
    fprintf('Mkdir %s !!!\n\n', outpath);
    mkdir(outpath);
end
if strcmp(problem, 'objectworld')
    outfname = sprintf('c%d_traj%dx%d_%s', ncolors, nsamples, samplelength, algorithm);
elseif strcmp(problem, 'gridworld')
    outfname = sprintf('b%d_traj%dx%d_%s', blocksize, nsamples, samplelength, algorithm);
else
    outfname = sprintf('traj%dx%d_%s', nsamples, samplelength, algorithm);
end
save(sprintf('%s/%s_test.mat', outpath, outfname), 'test_results', '-v7.3');
save(sprintf('%s/%s_transfer.mat', outpath, outfname), 'transfer_results', '-v7.3');
if ~isempty(fig)
    saveas(gcf, sprintf('%s/%s.fig', outpath, outfname));
    close(fig);
end

rmpaths;
