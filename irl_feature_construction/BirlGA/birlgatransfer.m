% Transfer learned reward function to a new state space.
function irl_result = birlgatransfer(prev_result, mdp_data, ~, ...
    feature_data, true_features, verbosity)

% Get weigths.
w = prev_result.model_itr{end};

F = buildfeaturematrix(mdp_data, feature_data, true_features, w);

% Compute reward.
r = repmat(F*w, 1, mdp_data.actions);

% Solve MDP.
soln = standardmdpsolve(mdp_data, r);
v    = soln.v;
q    = soln.q;
p    = soln.p;

% Build IRL result.
irl_result = struct('r', r, 'v', v, 'p', p, 'q', q, ...
    'r_itr', {{r}}, 'model_itr', {{w}}, ...
    'model_r_itr', {{r}}, 'p_itr', {{p}}, 'model_p_itr', {{p}}, ...
    'time', 0, 'score', 0);

end