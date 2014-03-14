% test IRL algorithms for a single expert
%
function hst = testSEIRL(problem, irlOpts)

hst = cell(length(problem.iters), 1);
for iter = problem.iters
    fprintf('## %d ##\n', iter);
    
    % generate data
    problem.seed = problem.initSeed + iter;
    mdp  = generateProblem(problem, problem.seed, problem.discount);
    data = generateDemonstration(mdp, problem);
    
    % IRL
    RandStream.setDefaultStream(RandStream.create('mrg32k3a', ...
        'NumStreams', 1, 'Seed', problem.seed));
    tic;
    [wL, logPost] = feval(irlOpts.alg, data.trajSet, [], mdp, irlOpts);
    elapsedTime = toc;
    
    % evaluate solution
    results = evalSEIRL(wL, data.weight, mdp);
    fprintf('- SEIRL results: [R] %f  [P] %f  [V] %f : %.2f sec\n\n', ...
        results.rewardDiff, results.policyDiff, results.valueDiff, elapsedTime);
    hst{iter}      = results;
    hst{iter}.wL   = wL;
    hst{iter}.data = data;
    hst{iter}.mdp  = mdp;
end

end