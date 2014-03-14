% Visualize the results such as priors, likelihood, and posterior
function fig = npbfirlvisualize(test_results)

ntrials   = length(test_results);
hstlength = length(test_results{1}.irl_result.hst.tm);

logPosts  = nan(hstlength, ntrials);
logLks    = nan(hstlength, ntrials);
logPrW    = nan(hstlength, ntrials);
logPrZ    = nan(hstlength, ntrials);
logPrT    = nan(hstlength, ntrials);
nU        = nan(hstlength, ntrials);
nF        = nan(hstlength, ntrials);
nZ        = nan(hstlength, ntrials);
sZ        = nan(hstlength, ntrials);
tms       = nan(hstlength, ntrials);
for i = 1:ntrials
    logPosts(:, i) = test_results{i}.irl_result.hst.logPost;
    logLks(:, i)   = test_results{i}.irl_result.hst.logLk;
    logPrW(:, i)   = test_results{i}.irl_result.hst.logPriorW;
    logPrZ(:, i)   = test_results{i}.irl_result.hst.logPriorZ;
    logPrT(:, i)   = test_results{i}.irl_result.hst.logPriorT;
    tms(:, i)      = test_results{i}.irl_result.hst.tm;
    for j = 1:hstlength
        nU(j, i) = nnz(test_results{i}.irl_result.hst.X{j}.T);
        nF(j, i) = length(test_results{i}.irl_result.hst.X{j}.w);
        nZ(j, i) = nnz(test_results{i}.irl_result.hst.X{j}.Z);
        sZ(j, i) = nZ(j, i)/numel(test_results{i}.irl_result.hst.X{j}.Z);
    end
end

X          = 0:hstlength - 1;
avgLogPost = mean(logPosts, 2);
avgLogLk   = mean(logLks, 2);
avgLogPrW  = mean(logPrW, 2);
avgLogPrZ  = mean(logPrZ, 2);
avgLogPrT  = mean(logPrT, 2);
avgNU      = mean(nU, 2);
avgNF      = mean(nF, 2);
avgNZ      = mean(nZ, 2);
avgSZ      = mean(sZ, 2);
seLogPost  = sqrt(var(logPosts, [], 2)/ntrials);
seLogLk    = sqrt(var(logLks, [], 2)/ntrials);
seLogPrW   = sqrt(var(logPrW, [], 2)/ntrials);
seLogPrZ   = sqrt(var(logPrZ, [], 2)/ntrials);
seLogPrT   = sqrt(var(logPrT, [], 2)/ntrials);
seNU       = sqrt(var(nU, [], 2)/ntrials);
seNF       = sqrt(var(nF, [], 2)/ntrials);
seNZ       = sqrt(var(nZ, [], 2)/ntrials);
seSZ       = sqrt(var(sZ, [], 2)/ntrials);

fig = figure;
subplot(5, 2, 1);
hold on;
plot(X, avgLogPost);
plot(X, avgLogPost + seLogPost, 'r:');
plot(X, avgLogPost - seLogPost, 'r:');
xlabel('iterations');
ylabel('log(posterior)');
ylim([min(avgLogPost - seLogPost), max(avgLogPost + seLogPost)]*0.5);
title(sprintf('NPB-FIRL on %s (%d trials)', test_results{1}.problem, ntrials));

subplot(5, 2, 3);
hold on;
plot(X, avgLogLk);
plot(X, avgLogLk + seLogLk, 'r:');
plot(X, avgLogLk - seLogLk, 'r:');
xlabel('iterations');
ylabel('log(likelihood)');
ylim([min(avgLogLk - seLogLk), max(avgLogLk + seLogLk)]*0.5);

subplot(5, 2, 5);
hold on;
plot(X, avgLogPrW);
plot(X, avgLogPrW + seLogPrW, 'r:');
plot(X, avgLogPrW - seLogPrW, 'r:');
xlabel('iterations');
ylabel('log(prior W)');

subplot(5, 2, 7);
hold on;
plot(X, avgLogPrZ);
plot(X, avgLogPrZ + seLogPrZ, 'r:');
plot(X, avgLogPrZ - seLogPrZ, 'r:');
xlabel('iterations');
ylabel('log(prior Z)');

subplot(5, 2, 9);
hold on;
plot(X, avgLogPrT);
plot(X, avgLogPrT + seLogPrT, 'r:');
plot(X, avgLogPrT - seLogPrT, 'r:');
xlabel('iterations');
ylabel('log(prior T)');

subplot(5, 2, 2);
hold on;
plot(X, avgNU);
plot(X, avgNU + seNU, 'r:');
plot(X, avgNU - seNU, 'r:');
xlabel('iterations');
ylabel('# of used base features');

subplot(5, 2, 4);
hold on;
plot(X, avgNF);
plot(X, avgNF + seNF, 'r:');
plot(X, avgNF - seNF, 'r:');
xlabel('iterations');
ylabel('# of generated features');

subplot(5, 2, 6);
hold on;
plot(X, avgNZ);
plot(X, avgNZ + seNZ, 'r:');
plot(X, avgNZ - seNZ, 'r:');
xlabel('iterations');
ylabel('# of non-zero entries in Z');

subplot(5, 2, 8);
hold on;
plot(X, avgSZ);
plot(X, avgSZ + seSZ, 'r:');
plot(X, avgSZ - seSZ, 'r:');
xlabel('iterations');
ylabel('sparsity of Z');

% INTERVAL = 0.5;
% minT = min(tms(:));
% maxT = max(tms(:));
% T    = minT: INTERVAL: maxT;
% nT   = length(T);
% logPost    = nan(nT, ntrials);
% logLk      = nan(nT, ntrials);
% avgLogPost = nan(nT, ntrials);
% avgLogLk   = nan(nT, ntrials);
% seLogPost  = nan(nT, ntrials);
% seLogLk    = nan(nT, ntrials);
% for p = 1:nT
%     tmpLogPost = nan(ntrials, 1);
%     tmpLogLk   = nan(ntrials, 1);
%     for i = 1:ntrials
%         q = find(tms(:, i) < T(p), 1, 'last');
%         if ~isempty(q)
%             tmpLogPost(i) = logPosts(q, i);
%             tmpLogLk(i)   = logLks(q, i);
%         end
%     end
%     cnt           = nnz(~isnan(tmpLogPost));
%     avgLogPost(p) = mean(tmpLogPost);
%     avgLogLk(p)   = mean(tmpLogLk);
%     seLogPost(p)  = sqrt(var(tmpLogPost)/cnt);
%     seLogLk(p)    = sqrt(var(tmpLogLk)/cnt);
% end
% 
% subplot(2, 2, 2);
% hold on;
% plot(T, avgLogLk);
% plot(T, avgLogLk + seLogLk, 'r:');
% plot(T, avgLogLk - seLogLk, 'r:');
% xlabel('CPU time (sec)');
% ylabel('log(likelihood)');
% if ~isempty(ylimLk)
%     ylim(ylimLk);
% end
% 
% subplot(2, 2, 4);
% hold on;
% plot(T, avgLogPost);
% plot(T, avgLogPost + seLogPost, 'r:');
% plot(T, avgLogPost - seLogPost, 'r:');
% xlabel('CPU time (sec)');
% ylabel('log(posterior)');
% if ~isempty(ylimPost)
%     ylim(ylimPost);
% end