% Construct the raw features for the highway domain.
function [feature_data, r] = highwayfeatures2(mdp_params, mdp_data)

% mdp_params - definition of MDP domain
% mdp_data - generic definition of domain
% feature_data - generic feature data:
%   splittable - matrix of states to features
%   stateadjacency - sparse state adjacency matrix

if mdp_params.continuous
    error('Cannot build the continuous version.');
end

% Fill in default parameters.
mdp_params = highwaydefaultparams(mdp_params);
nS         = mdp_data.states;
nA         = mdp_data.actions;
length     = mdp_params.length;
nSpeeds    = mdp_params.speeds;
nLanes     = mdp_params.lanes;
nC1        = mdp_params.c1;
nC2        = mdp_params.c2;
maxdist    = mdp_params.maxdist;
nDists     = 2*maxdist + 1;

% Construct adjacency table.
stateadjacency = sparse([], [], [], nS, nS, nS*nA);
for s = 1:nS
    for a = 1:nA
        stateadjacency(s, mdp_data.sa_s(s, a, 1)) = 1;
    end;
end;

% Construct split table.
% Features are the following:
% 1..speeds - indicates current speed.
% 1..lanes - indicates current lane.

% 1..(nC1 + nC2)*(-maxdist:maxdist) ...
% : object of type X is at most N car-lengths in front/behind me
tests      = nSpeeds + nLanes + (nC1 + nC2)*nDists;
splittable = zeros(nS, tests);
for s = 1:nS
    [x, lane, speed] = highwaystatetocoord(s, mdp_params);

    % Write lane and speed features.
    splittable(s, 1:nSpeeds) = [ones(1, speed), zeros(1, nSpeeds - speed)];
    splittable(s, nSpeeds + lane) = 1;

    % Check close objects
    tmp = nSpeeds + nLanes;
    for c = 1:(nC1 + nC2)
        for d1 = 0:maxdist
            for d2 = 0:d1
                x1 = mod(x + d2 - 1, length) + 1;
                for y = 1:nLanes
                    if (c > nC1 && mdp_data.map1(x1, y) == c - nC1) ...
                            || (c <= nC1 && mdp_data.map1(x1, y) == c)
                        fi = (c - 1)*nDists + d1 + 1;
                        splittable(s, tmp + fi) = 1;
                    end
                end
            end
        end
        for d1 = -maxdist:-1
            for d2 = d1:-1
                x1 = mod(x + d2 - 1, length) + 1;
                for y = 1:nLanes
                    if (c > nC1 && mdp_data.map1(x1, y) == c - nC1) ...
                            || (c <= nC1 && mdp_data.map1(x1, y) == c)
                        fi = (c - 1)*nDists + nDists + d1 + 1;
                        splittable(s, tmp + fi) = 1;
                    end
                end
            end
        end
    end
end

% % 1..(nC1 + nC2)*(-maxdist:maxdist)*nLanes ...
% % : object of type X is at most N car-lengths on Y lane in front/behind me
% tests      = nSpeeds + nLanes + (nC1 + nC2)*nDists*nLanes;
% splittable = zeros(nS, tests);
% for s = 1:nS
%     [x, lane, speed] = highwaystatetocoord(s, mdp_params);
%     
%     % Write lane and speed features.
%     splittable(s, 1:nSpeeds) = [ones(1, speed), zeros(1, nSpeeds - speed)];
%     splittable(s, nSpeeds + lane) = 1;
%     
%     % Check close objects
%     tmp = nSpeeds + nLanes;
%     for y = 1:nLanes
%         for c = 1:(nC1 + nC2)
%             for d1 = 0:maxdist % forward direction
%                 for d2 = 0:d1
%                     x1 = mod(x + d2 - 1, length) + 1;
%                     if (c > nC1 && mdp_data.map1(x1, y) == c - nC1) ...
%                             || (c <= nC1 && mdp_data.map1(x1, y) == c)
%                         fi = (c - 1)*nDists*nLanes + d1*nLanes + y;
%                         splittable(s, tmp + fi) = 1;
%                     end
%                 end
%             end
%             for d1 = -maxdist:-1 % backward direction
%                 for d2 = d1:-1
%                     x1 = mod(x + d2 - 1, length) + 1;
%                     if (c > nC1 && mdp_data.map1(x1, y) == c - nC1) ...
%                             || (c <= nC1 && mdp_data.map1(x1, y) == c)
%                         fi = (c - 1)*nDists*nLanes + (nDists + d1)*nLanes + y;
%                         splittable(s, tmp + fi) = 1;
%                     end
%                 end
%             end
%         end
%     end
% end

% Return feature data structure.
feature_data = struct('stateadjacency', stateadjacency, 'splittable', splittable);

% Contruct true reward function.
r = zeros(nS, 1);
if strcmp(mdp_params.policy_type, 'outlaw')
    dist = 2;
    for s = 1:nS
        [x, lane, speed] = highwaystatetocoord(s, mdp_params);
        nearPolice = false;
        
        % Check close police
        for y = 1:nLanes
            for d = -dist:dist
                xd = mod(x + d - 1, length) + 1;
                if mdp_data.map1(xd, y) == 1 % is police
                    nearPolice = true;
                    break;
                end
            end
        end
        
        if nearPolice
            if speed > 2
                r(s) = -20; %-10; %-20;
            elseif speed > 1
                r(s) = -5; %0; %-5;
            else
                r(s) = -10; %-2; %-10;
            end
        else
            if speed > 3
                r(s) = 10; %6; %10;
            elseif speed > 2
                r(s) = 5; %2; %5;
            elseif speed > 1
                r(s) = -5; %0; %-5;
            else
                r(s) = -10; %-2; %-10;
            end
        end
    end
elseif strcmp(mdp_params.policy_type, 'lawful')
    error('Reward for lawful policy is not yet implemented.');
    for s = 1:nS
        [x, lane, speed] = highwaystatetocoord(s, mdp_params);
        if speed > 3
            if lane > 2
                r(s) = -10;
            elseif lane > 1
                r(s) = -5;
            else
                r(s) = 0;
            end
        elseif speed > 2
            if lane > 1
                r(s) = -5;
            else
                r(s) = 0;
            end
        else
            r(s) = 0;
        end
    end
end

r = sparse(repmat(r, 1, nA));
