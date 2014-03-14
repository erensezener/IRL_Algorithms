% Get a good set of inducing states under the specified parameters.
function gp = gpirlgetinducingpoints(gp,~,mu_sa,algorithm_params)

% Constants.
states = size(mu_sa,1);
mu_s = sum(mu_sa,2);

if strcmp(algorithm_params.inducing_pts,'examples'),
    % Select just the example states.
    s_u = find(mu_s)';
    
elseif strcmp(algorithm_params.inducing_pts,'limitedexamples'),
    nS   = length(mu_s);
    pts  = find(mu_s)';
    nPts = ceil(nS*0.2);
    if length(pts) > nPts
        fprintf('Inducing points are chosen by visitation count (%d/%d)\n', ...
            nPts, length(pts));
        [~, ix] = sort(mu_s, 'descend');
        pts     = ix(1:nPts);
%     else
%         fprintf('Add randomly chosen inducing points (%d/%d)\n', ...
%             nPts - length(pts), nPts);
%         otherPts  = find(mu_s == 0)';
%         ix        = randperm(length(otherPts));
%         nOtherPts = nPts - length(pts);
%         pts       = horzcat(pts, otherPts(ix(1:nOtherPts)));
    end
    s_u = pts;
elseif strcmp(algorithm_params.inducing_pts,'examplesplus'),
    % Select example states, plus random states to bring up to desired
    % total.
    s_u = find(mu_s)';
    if length(s_u) < algorithm_params.inducing_pts_count,
        other_states = find(~mu_s)';
        other_states = other_states(randperm(length(other_states)));
        s_u = [s_u other_states(...
            1:(algorithm_params.inducing_pts_count-length(s_u)))];
        s_u = sort(s_u);
    end;
elseif strcmp(algorithm_params.inducing_pts,'random'),
    % Select random states.
    s_u = randperm(states);
    s_u = sort(s_u(1:algorithm_params.inducing_pts_count));
else
    % Select all states.
    s_u = 1:states;
end;

% Set inducing points on the gp.
gp.s_u = s_u;
gp.X_u = gp.X(s_u,:);
