% Maximum-a-posterior inference for Bayesian IRL using gradient ascent
% Written by Jaedeug Choi
function irl_result = birlgarun(alg_params, mdp_data, ~, ...
    feature_data, example_samples, true_features, verbosity)

% alg_params: parameters of the BIRL using gradient ascent algorithm:
%   seed (0)          - initialization for random seed
%   eta (1.0)         - confidence parameter of choosing optimal actions
%   normalPrior (1)   - use zero mean normal prior for regularization
%   laplacePrior (0)  - use zero mean Laplace prior for regularization
%   sigma (1.0)       - standard deviation of prior
%   nrestarts (5)     - # of random restarts
%   true_features (0) - use true features as a basis
%   all_features (1)  - use the provided features as a basis
% mdp_data:        definition of the MDP to be solved
% example_samples: cell array containing examples
% irl_result:      result of IRL algorithm, generic and algorithm-specific:
%   r    - inferred reward function
%   v    - inferred value function.
%   q    - corresponding q function.
%   p    - corresponding policy.
%   time - total running time

% Fill in default parameters.
alg_params = birlgadefaultparams(alg_params);

setrandomseed(alg_params.seed);

% Build state transition matrix
[nS, nA, ~] = size(mdp_data.sa_p);
mdp_data.T  = sparse(nS*nA, nS);
for a = 1:nA
    for s = 1:nS
        i = (a - 1)*nS + s;
        for k = 1:length(mdp_data.sa_s(s, a, :))
            s2 = mdp_data.sa_s(s, a, k);
            mdp_data.T(i, s2) = mdp_data.T(i, s2) + mdp_data.sa_p(s, a, k);
        end
    end
end

% Build feature membership matrix.
mdp_data.F = buildfeaturematrix(mdp_data, feature_data, true_features, alg_params);
nF         = size(mdp_data.F, 2);

% Count state-action vistation in example samples
[N, T]   = size(example_samples);
smpl.cnt = zeros(nS, nA);
for i = 1:N
    for t = 1:T
        s = example_samples{i,t}(1);
        a = example_samples{i,t}(2);
        smpl.cnt(s, a) = smpl.cnt(s, a) + 1;
    end
end
[i, j, k]  = find(smpl.cnt);
smpl.spcnt = [i, j, k];     

% Set up optimization options.
options         = struct();
options.Display = 'iter';
options.LS_init = 2;
options.LS      = 2;
options.Method  = 'lbfgs';
if verbosity == 0
    options.Display = 'none';
end

if verbosity ~= 0
    fprintf('BIRL-GA starts optimization.\n');
end

% Run unconstrainted non-linear optimization.
objFun = @(w) calneglogpost(w, smpl, mdp_data, alg_params);
tic;

ws = [];
fs = [];
for iter = 1:alg_params.nrestarts
    % Initialize reward.
    w = initw(nF, alg_params);
    
    [w, f] = minFunc(objFun, w, options);
    
    ws = horzcat(ws, w);
    fs = horzcat(fs, f);
end
[~, i] = min(fs);
minw   = ws(:, i);

% Print timing.
time = toc;
if verbosity ~= 0,
    fprintf('Optimization completed in %f seconds.\n', time);
end

% Convert to full tabulated reward.
r = mdp_data.F*minw;

% Return corresponding reward function.
r    = repmat(r, 1, nA);
soln = standardmdpsolve(mdp_data, r);
v    = soln.v;
q    = soln.q;
p    = soln.p;

% Construct returned structure.
irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, 'ws', ws, 'fs', fs, ...
    'r_itr', {{r}}, 'model_itr', {{minw}}, ...
    'model_r_itr', {{r}}, 'p_itr', {{p}}, 'model_p_itr', {{p}}, ...
    'time', time);

end


%% Calculate negative log posterior in order to perform gradient descent
function [logPost, dLogPost, logPrior, logLk] ...
    = calneglogpost(w, smpl, mdp_data, alg_params)

if nargout > 1
    [logLk, dLogLk]       = calloglk(w, smpl, mdp_data, alg_params.eta);
    [logPrior, dLogPrior] = callogprior(w, alg_params);
    dLogPost              = -(dLogLk + dLogPrior);
else    
    logLk    = calloglk(smpl, w, mdp_data, alg_params.eta);
    logPrior = callogprior(w, alg_params);
end
logPost = -(logPrior + logLk);

if isinf(logPost) || isnan(logPost)
    error('ERROR in caluation of log posterior - prior: %f, llh:%f, ETA:%f, w:%f %f \n', ...
        logPrior, logLk, full(min(w)), full(max(w)));
end

end


%% Calculate log likelihood
function [logLk, dLogLk] = calloglk(w, smpl, mdp_data, ETA)

nS    = mdp_data.states;
nA    = mdp_data.actions;
nF    = size(w, 1);

% Solve MDP with given features and weight
mdpSoln = standardmdpsolve(mdp_data, repmat(mdp_data.F*w, 1, nA));
Q       = ETA.*mdpSoln.q;
Q       = bsxfun(@minus, Q, max(Q, [], 2));
nQ      = log(sum(exp(Q), 2));
logStP  = bsxfun(@minus, Q, nQ);
logLk   = sum(sum(logStP.*smpl.cnt));

if nargout > 1
    % compute soft-max policy
    stP = exp(logStP);
    
    % calculate dLogLk/dR
    dQ    = calgradq(mdpSoln.p, mdp_data)*mdp_data.F;
    dLogP = zeros(nS*nA, nF);
    for f = 1:nF
        x           = reshape(dQ(:, f), nS, nA);
        y           = sum(stP.*x, 2);
        z           = ETA.*bsxfun(@minus, x, y);
        dLogP(:, f) = reshape(z, nS*nA, 1);
    end
    
    % calculate gradient of reward function
    dLogLk = zeros(1, nF);
    for i = 1:size(smpl.spcnt, 1)
        s      = smpl.spcnt(i, 1);
        a      = smpl.spcnt(i, 2);
        n      = smpl.spcnt(i, 3);
        dLogLk = dLogLk + n*dLogP((a - 1)*nS + s, :);
    end
    dLogLk = dLogLk';
end

end


%% Calculate gradient of Q-function
function dQ = calgradq(p, mdp_data)

% p (|S|x1)      : policy
% dQ (|S||A|x|S|): dQ(s',a')/dR(s) = dQ((s',a'), (s))

nS = mdp_data.states;
nA = mdp_data.actions;

% calculate dQ/dw
Eplc = sparse(1:nS, (p - 1)*nS + (1:nS)', ones(nS, 1), nS, nS*nA);
T    = mdp_data.discount.*mdp_data.T*Eplc;
F    = repmat(speye(nS, nS), nA, 1);
dQ   = (speye(nS*nA) - T)\F;

% MAX_ITERS = 10^4;
% EPS       = 1e-4;
% dQ        = F;
% for iter = 1:MAX_ITERS
%     oldDQ = dQ;
%     dQ    = F + T*oldDQ;
%     if norm(dQ(:) - oldDQ(:), 'inf') < EPS, break; end
% end

end


%% Calculate log prior of reward
function [logPrior, dLogPrior] = callogprior(w, alg_params)

x = w;
s = alg_params.sigma;

if alg_params.normal_prior          % normal prior
    logPrior = -(x'*x)/(2*s^2);
elseif alg_params.laplace_prior     % laplace prior
    logPrior = -sum(abs(x));
else                                % Uuiform prior
    logPrior = 0;
end

if nargout > 1
    if alg_params.normal_prior      % normal prior
        dLogPrior = -x./s^2;
    elseif alg_params.laplace_prior % laplace prior
        dLogPrior = -sign(x);
    else                            % Uuiform prior
        dLogPrior = zeros(size(x));
    end
end

end


%% Initialize weight
function w = initw(N, alg_params)

if alg_params.normal_prior          % normal prior
    w = normrnd(0, alg_params.sigma, N, 1);
elseif alg_params.laplace_prior     % laplace prior
    w = rand(nF, 1);
else                                % Uuiform prior
    w = rand(N, 1);
end

end

