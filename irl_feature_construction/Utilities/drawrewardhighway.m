% Draw single highway with specified reward function.
function drawrewardhighway(r, mdp_params, mdp_data, y1, y2)

% Set up the axes.
lanes  = mdp_params.lanes;
speeds = mdp_params.speeds;
length = mdp_params.length;
axis([0, lanes*speeds, 0, length]);
set(gca, 'xtick', []);
set(gca, 'ytick', []);
daspect([1 1 1]);

% Colors.
curb_color    = [0.35 0.5 0.25];
road_color    = [0.35 0.37 0.4];

if ~isempty(r)
    % Draw reward.
    maxr = max(max(r));
    minr = min(min(r));
    rngr = maxr-minr;
    for y = 1:length
        for lane = 1:lanes
            for spd = 1:speeds
                x = (spd - 1)*lanes + lane;
                s = highwaycoordtostate(y, lane, spd, mdp_params);
                % Draw reward.
                if rngr == 0
                    v = 0.0;
                else
                    v = (mean(r(s, :), 2) - minr)/rngr;
                end
                color = [v v v];
                color = min(ones(1, 3), max(zeros(1, 3), color));
                patch([x - 1, x - 1, x, x], [y - 1, y, y, y - 1], color, 'EdgeColor', 'none');
            end
        end
    end
end

% Draw curbs.
line([0 0], [0 length], 'linewidth', 4, 'color', curb_color);
% Write speed
if nargin <= 3
    y1 = 1;
end
% text(-2.7, y1 - 1.7, 'speed: ');
for spd = 1:speeds
    % Draw the road.
    for lane = 1:lanes
        x = (spd - 1)*lanes + lane;
        line([x x], [0 length], 'linestyle', '--', ...
            'linewidth', 1, 'color', curb_color);
    end
    % Draw curbs.
    line([spd*mdp_params.lanes spd*mdp_params.lanes], [0 length], ...
        'linewidth', 4, 'color', curb_color);
    % Write speed
    text((spd)*lanes - 1.7, y1 - 1.7, sprintf('%d', spd));    
end

% Draw the cars.
for spd = 1:speeds
    for i = 1:size(mdp_data.c1array, 1)
        for j = 1:size(mdp_data.c1array{i}, 1)
            % Get colors and position of object.
            lane = mdp_data.c1array{i}(j, 2);
            x    = (spd - 1)*lanes + lane;
            y    = mdp_data.c1array{i}(j, 1);
            c1   = i;
            c2   = mdp_data.map2(y, lane);
            
            % Draw the car.
            highwayrenderobject(x, y, c1, c2, true);
        end
    end
end

% % Draw the reward function.
% for x = 1:length
%     for lane = 1:lanes
%         for spd = 1:speeds
%             xpos = (spd - 1)*lanes + lane;
%             ypos = x;
%             s    = highwaycoordtostate(x, lane, spd, mdp_params);
%             if rngr == 0
%                 v = 0.0;
%             else
%                 v = (mean(r(s, :), 2) - minr)/rngr;
%             end
%             color = [v v v];
%             color = min(ones(1, 3), max(zeros(1, 3), color));
%             rectangle('Position', [xpos - 1, ypos - 1, 1, 1], 'FaceColor', color);
%         end
%     end
% end
% 
% % Convert p to action mode.
% if size(p, 2) ~= 1
%     [~, p] = max(p, [], 2);
% end
% 
% % Draw delimiting markers for speeds.
% for spd = 0:speeds
%     line([spd*mdp_params.lanes spd*mdp_params.lanes], [0 length], ...
%         'linewidth', 2, 'color', 'b');
% end
% 
% % Action highway to gridworld conversion table.
% actionmap = [ 5 3 4 2 1 ];
% 
% if ~isempty(p)
%     % Draw paths.
%     for x = 1:length
%         for lane = 1:lanes
%             for spd = 1:speeds
%                 xpos = (spd - 1)*lanes + lane;
%                 ypos = x;
%                 s    = highwaycoordtostate(x, lane, spd, mdp_params);
%                 a    = p(s);                
%                 a    = actionmap(a); % Convert a from highway action to gridworld action.
%                 gridworlddrawagent(xpos, ypos, a, [0, 1, 0]);
%             end
%         end
%     end
% end
% 
% % Initialize colors.
% shapeColors = colormap(lines(mdp_params.c1 + mdp_params.c2));
% 
% % Draw objects.
% for i = 1:size(mdp_data.c1array, 1)
%     for j = 1:size(mdp_data.c1array{i}, 1)
%         % Get colors and position of object.
%         y    = mdp_data.c1array{i}(j, 1);
%         lane = mdp_data.c1array{i}(j, 2);
%         c1   = i;
%         c2   = mdp_data.map2(y, lane);
%         
%         for spd = 1:speeds
%             % Draw the object.
%             x = (spd - 1)*lanes + lane;
%             rectangle('Position', [x - 0.65, y - 0.65, 0.3, 0.3], ...
%                 'Curvature', [1, 1], ...
%                 'FaceColor', shapeColors(c1, :), ...
%                 'EdgeColor', shapeColors(c1, :), 'LineWidth', 1);
%         end
%     end
% end
% 
% for c1 = 1:mdp_params.c1
%     for c2 = 1:mdp_params.c2
%         rectangle('Position', [c1, -c2, 0.3, 0.3], ...
%             'Curvature', [1, 1], ...
%             'FaceColor', shapeColors(c1, :), ...
%                 'EdgeColor', shapeColors(c1, :), 'LineWidth', 1);
%     end
% end

if nargin > 3
    axis([0, lanes*speeds, y1 - 1, y2]);
end
