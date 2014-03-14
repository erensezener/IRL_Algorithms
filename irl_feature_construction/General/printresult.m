% Print pre-computed IRL test result
% Modified by Jaedeug Choi
function printresult(test_result)

% Print results.
for k = 1:length(test_result)
    alg     = test_result(k).algorithm;
    problem = test_result(k).problem;
    if length(test_result) ~= 1
        if k == 1
            fprintf('Printing results for processed version:\n');
        else
            fprintf('Printing results for non-processed version:\n');
        end
    end
    for i = 1:length(test_result(k).test_models)
        world = test_result(k).test_models{i};
        for j = 1:length(test_result(k).test_metrics)
            if isfield(test_result(k).metric_scores, 'map') ...
                    && isfield(test_result(k).metric_scores, 'mean')
                fprintf('[MAP ] %s on %s, %s %s: ', alg, problem, world, ...
                    test_result(k).test_metrics{j});
                printmetric(getmetric(test_result, k, i, j, 'map'));
                
                fprintf('[Mean] %s on %s, %s %s: ', alg, problem, world, ...
                    test_result(k).test_metrics{j});
                printmetric(getmetric(test_result, k, i, j, 'mean'));
            else
                fprintf('%s on %s, %s %s: ', alg, problem, world, ...
                    test_result(k).test_metrics{j});
                printmetric(getmetric(test_result, k, i, j, []));
            end
        end
    end
end

end


function printmetric(metric)

if length(metric) == 1
    fprintf('%10.4f\n', metric);
elseif length(metric) == 2
    fprintf('%10.4f (%10.4f)\n', metric(1), metric(2));
else
    fprintf('%10.4f (%10.4f vs %10.4f)\n', metric(1), metric(2), metric(3));
end

end


function metric = getmetric(test_result, k, i, j, type)

if strcmp(type, 'map')
    metric = test_result(k).metric_scores.map{i, j};
elseif strcmp(type, 'mean')
    metric = test_result(k).metric_scores.mean{i, j};
else
    metric = test_result(k).metric_scores{i, j};
end

end
