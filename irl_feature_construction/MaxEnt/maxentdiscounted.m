% Compute MaxEnt objective and gradient. Discounted infinite-horizon version.
function [val,dr] = maxentdiscounted(r,F,muE,mu_sa,mdp_data,initD,laplace_prior)

if nargin < 7,
    laplace_prior = 0;
end;

% Compute constants.
actions = mdp_data.actions;

% Convert to full reward.
wts = r;
r = F*r;

[~,~,policy,logpolicy] = linearvalueiteration(mdp_data,repmat(r,1,actions));

% Compute value by adding up log example probabilities.
val = sum(sum(logpolicy.*mu_sa));

% Add laplace prior.
if laplace_prior,
    val = val - laplace_prior*sum(abs(wts));
end;

% Invert for descent.
val = -val;

if nargout >= 2,    
    % Compute state visitation count D.
    D = linearmdpfrequency(mdp_data,struct('p',policy),initD);

    % Compute gradient.
    dr = muE - F'*D;
    
    % Laplace prior.
    if laplace_prior,
        dr = dr - laplace_prior*sign(wts);
    end;
    
    % Invert for descent.
    dr = -dr;
end;

end


% % Compute the occupancy measure of the linear MDP given a policy.
% function D = linearmdpfrequency(mdp_data, mdp_solution, initD)
% 
% [states, actions, transitions] = size(mdp_data.sa_p);
% VITR_THRESH = 1e-4;
% %VITR_THRESH = 1e-10;
% 
% D = zeros(states, actions);
% 
% if nargin < 3 || isempty(initD)
%     % Initialize uniform initial state distribution.    
%     d = zeros(states, actions);
%     d(sum(mdp_data.sa_p, 3) > 0) = 1;
%     initD = (1/states)*bsxfun(@rdivide, d, sum(d, 2));
% else
%     d = zeros(states, actions);
%     d(sum(mdp_data.sa_p, 3) > 0) = 1;
%     d = bsxfun(@rdivide, d, sum(d, 2));    
%     initD = d.*repmat(initD, [1 actions]);
% end
% 
% diff = 1.0;
% while diff >= VITR_THRESH
%     Dp = D;
%     Dpi = repmat(mdp_solution.p, [1 1 transitions]).*mdp_data.sa_p.* ...
%         repmat(sum(Dp, 2), [1 actions transitions])*mdp_data.discount;
%     
%     for a = 1:actions
%         x = mdp_data.sa_s(:, a, :);
%         y = Dpi(:, a, :);
%         z = sparse(x(:), 1:states*transitions, y(:), states, states*transitions);
%         D(:, a) = initD(:, a) + sum(z, 2);
%     end
%     diff = max(abs(D - Dp));
% end
% 
% D = sum(D, 2);
% 
% end
