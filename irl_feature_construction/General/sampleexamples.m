% Sample example tranjectories from the state space of a given MDP.
% Modified by Jaedeug Choi
function example_samples = sampleexamples(mdp_model, mdp_data, mdp_solution, test_params)

setrandomseed(test_params.seed);

% Allocate training samples.
N = test_params.training_samples;
T = test_params.training_sample_lengths;
example_samples = cell(N, T);
exp_values = zeros(N, 1);

% Sample trajectories.
for i = 1:N
    % Sample initial state.
    if isfield(mdp_data, 'b0')
        s = find(cumsum(mdp_data.b0) > rand, 1, 'last');
    else
        s = ceil(rand(1, 1)*mdp_data.states);
    end
    
    % Run sample trajectory.
    for t = 1:T
        % Compute optimal action for current state.
        a = samplingaction(mdp_data, mdp_solution, s, test_params);
        
        exp_values(i) = exp_values(i) + mdp_data.discount^(t - 1)*mdp_data.r(s, a);
        
        % Store example.
        example_samples{i, t} = [s; a];
        
        % Move on to next state.
        s = feval(strcat(mdp_model, 'step'), mdp_data, mdp_solution, s, a);
    end
end

fprintf('empirical value: %8.4f %8.4f : %8.4f\n', ...
    mean(exp_values), sqrt(var(exp_values)/length(exp_values)), ...
    mean(mdp_solution.v));

end


function action = samplingaction(mdp_data, mdp_solution, s, test_params)

% mdp_solution
%   p(|S|x1)  : policy
%   v(|S|x1)  : value function
%   q(|S|x|A|): Q-value function

if strcmp(test_params.type, 'allpossiblepolicies')
    q          = mdp_solution.q(s, :);
    maxQ       = max(q);
    actionList = find(maxQ - q < 1e-8);
    actionId   = randi(length(actionList));
    action     = actionList(actionId);
elseif strcmp(test_params.type, 'epsgreedy')
    if rand < test_params.epsilon
        action = randi(mdp_data.actions);
    else
        action = mdp_solution.p(s);
    end
elseif strcmp(test_params.type, 'random')
    action = randi(mdp_data.actions);
elseif strcmp(test_params.type, 'optimal')
    action = mdp_solution.p(s);
else
    warning('no type of sampling examples is defined');
    action = mdp_solution.p(s);
end

end