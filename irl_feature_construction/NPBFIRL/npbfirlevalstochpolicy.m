function soln = npbfirlevalstochpolicy(X, mdp_data, p)

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

[v, q] = evaluate(p, r, mdp_data);
soln = struct('v', v, 'q', q, 'p', p);

end



%% Evaluate policy on MDP with given reward
function [v, q] = evaluate(p, r, mdp_data)

% Evaluate policy measured on the given reward
% V = (I-gamma*T^piL)^-1 R^piL
%
nS = mdp_data.states;
nA = mdp_data.actions;

Tp = sparse(nS, nS);
for a = 1:nA
    Tp = Tp + bsxfun(@times, mdp_data.TG{a}(:, :), p(:, a));
end

if size(r, 2) == 1
    v = (speye(nS) - Tp)\r;
    q = mdp_data.discount*sum(mdp_data.sa_p.*v(mdp_data.sa_s), 3);
    q = bsxfun(@plus, q, r);
elseif size(r, 2) == nA
    Rp = sparse(nS, 1);
    for a = 1:nA
        Rp = Rp + p(:, a).*r(:, a);
    end
    v = (speye(nS) - Tp)\Rp;
    q = r + mdp_data.discount*sum(mdp_data.sa_p.*v(mdp_data.sa_s), 3);
else
    error('The dimension of reward is not correct.');
end

end
