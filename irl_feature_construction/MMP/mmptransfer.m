% Transfer learned reward function to a new state space.
function irl_result = mmptransfer(prev_result,mdp_data,mdp_model,...
    feature_data,true_feature_map,verbosity)

% Get weigths.
wts = prev_result.model_itr{end};

F = buildfeaturematrix(mdp_data, feature_data, true_feature_map, wts);

% Compute reward.
r = repmat(F*wts,1,mdp_data.actions);

% Solve MDP.
soln = feval([mdp_model 'solve'],mdp_data,r);
v = soln.v;
q = soln.q;
p = soln.p;

% Build IRL result.
irl_result = struct('r',r,'v',v,'p',p,'q',q,'model_itr',{{wts}},...
    'r_itr',{{r}},'model_r_itr',{{r}},'p_itr',{{p}},'model_p_itr',{{p}},...
    'time',0,'score',0);
