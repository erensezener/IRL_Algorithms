% Solve MDP using previous solution
function soln = npbfirlsolvemdp(X, mdp_data, prevSoln, reusev)

if nargin < 4 || isempty(reusev)
    reusev = false;
end

if size(X, 1) == mdp_data.states
    r = X;
else
    if ~isfield(X, 'F') || isempty(X.F)
        F = npbfirlgenfeatmtrx(X, mdp_data.F);
    else
        F = X.F;
    end
    r = F*X.w;
end

shouldSolveMdp = true;
if ~isempty(prevSoln)    
    [vn, qn] = evaluate(prevSoln.p, r, mdp_data);
    [~, pn]  = max(qn, [], 2);
    if isequal(prevSoln.p, pn)
        shouldSolveMdp = false;
        soln           = prevSoln;
        soln.v         = vn;
        soln.q         = qn;
    end
else
    pn = [];
    vn = [];
end

if shouldSolveMdp
%     r = repmat(r, 1, mdp_data.actions);
%     if reusev
%         v = stdvalueiteration(mdp_data, r, vn);        
%     else
%         v = stdvalueiteration(mdp_data, r);
%     end
%     [q, p] = stdpolicy(mdp_data, r, v);
    if reusev
        [p, v, q] = policyiteration(mdp_data, r, pn, vn);
    else
        [p, v, q] = policyiteration(mdp_data, r, [], []);
    end
    soln = struct('v', v, 'q', q, 'p', p);
end

end


%% Policy iteration
function [p, v, q] = policyiteration(mdp_data, r, pinit, vinit)

diff = 1;

if ~isempty(pinit) && ~isempty(vinit)
    p = pinit;
    v = vinit;
else    
    p = ones(mdp_data.states, 1);
    v = zeros(mdp_data.states, 1);
end

t = 0;
while diff > 0
    pn     = p;
    vn     = v;
    [v, q] = evaluate(pn, r, mdp_data);
    [~, p] = max(q, [], 2);
    diff   = nnz(p ~= pn);
    
    if diff > 0 && t > 20
        diff = max(abs(v - vn)) > 1e-8;
    end
    t = t + 1;
end

end
    


%% Evaluate policy on MDP with given reward
function [v, q] = evaluate(p, r, mdp_data)

[nS, nA, ~] = size(mdp_data.sa_p);

Tp = sparse(nS, nS);
for a = 1:nA
    ix = find(p == a);
    if ~isempty(ix)
        Tp(ix, :) = mdp_data.TG{a}(ix, :);
    end
end

if size(r, 2) == 1
    v = (speye(nS) - Tp)\r;
    q = sum(mdp_data.sa_p.*v(mdp_data.sa_s), 3)*mdp_data.discount;
    q = bsxfun(@plus, q, r);
elseif size(r, 2) == mdp_data.actions
    v = (speye(nS) - Tp)\mean(r, 2);
    q = r + sum(mdp_data.sa_p.*v(mdp_data.sa_s), 3)*mdp_data.discount;
else
    error('The dimension of reward is not correct.');
end

end
