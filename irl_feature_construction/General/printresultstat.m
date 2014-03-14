% Print the statistics of the results
% Written by Jaedeug Choi
function [scores1, scores2] = printresultstat(test_results, transfer_results, type, outfname)

ntrials    = length(test_results);
ntransfers = size(transfer_results, 2);

if strcmp(type, 'map')
    nscores = length(test_results{1}.metric_scores.map);
elseif strcmp(type, 'mean')
    nscores = length(test_results{1}.metric_scores.mean);
else
    nscores = length(test_results{1}.metric_scores);
end

scores1 = nan(nscores, ntrials);
scores2 = nan(nscores, ntrials, ntransfers);
for i = 1:ntrials
    for j = 1:nscores
        if strcmp(type, 'map')
            scores1(j, i) = test_results{i}.metric_scores.map{j}(1);
            for k = 1:ntransfers
                scores2(j, i, k) = transfer_results{i, k}.metric_scores.map{j}(1);
            end
        elseif strcmp(type, 'mean')
            scores1(j, i) = test_results{i}.metric_scores.mean{j}(1);
            for k = 1:ntransfers
                scores2(j, i, k) = transfer_results{i, k}.metric_scores.mean{j}(1);
            end
        else
            scores1(j, i) = test_results{i}.metric_scores{j}(1);
            for k = 1:ntransfers
                scores2(j, i, k) = transfer_results{i, k}.metric_scores{j}(1);
            end
        end
    end
end
avgScore1 = mean(scores1, 2);
avgScore2 = mean(mean(scores2, 3), 2);
seScore1  = sqrt(var(scores1, [], 2)/ntrials);
seScore2  = sqrt(var(mean(scores2, 3), [], 2)/ntrials);

alg     = test_results{1}.algorithm;
problem = test_results{1}.problem;
metrics = test_results{1}.test_metrics;
if strcmp(type, 'map')
    fprintf('## [MAP ] %s on %s (%d trials) ##\n', alg, problem, ntrials);
elseif strcmp(type, 'mean')
    fprintf('## [Mean] %s on %s (%d trials) ##\n', alg, problem, ntrials);
else
    fprintf('## %s on %s (%d trials) ##\n', alg, problem, ntrials);
end
for i = 1:nscores
    fprintf('%s\n', metrics{i});
    fprintf(' - training: %8.4f (%8.4f)\n', avgScore1(i), seScore1(i));
    fprintf(' - transfer: %8.4f (%8.4f)\n', avgScore2(i), seScore2(i));
end
fprintf('\n');

if ~isempty(outfname)
    for i = 1:nscores
        outfname2 = sprintf('%s_%s.txt', outfname, metrics{i});
        fid = fopen(outfname2, 'a+');
        fprintf(fid, '%s\t%12.4f\t%12.4f\t%12.4f\t%12.4f\t%s\n', ...
            alg, avgScore1(i), seScore1(i), avgScore2(i), seScore2(i), datestr(now, 'yymmdd_HH:MM:SS'));
        fclose(fid);
    end
end

end