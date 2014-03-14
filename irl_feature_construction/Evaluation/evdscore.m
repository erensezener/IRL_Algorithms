% Compute the expected value difference
function score = evdscore(mdpSoln, reward, irlSoln, ~, ~, ~, ...
    mdpData, ~, model)

% Compute occupancies measures for IRL result and true policy.
omIrl = feval(strcat(model, 'frequency'), mdpData, irlSoln);
omMdp = feval(strcat(model, 'frequency'), mdpData, mdpSoln);

% Compute expected value
expValIrl = omIrl'*mean(reward, 2);
expValMdp = omMdp'*mean(reward, 2);

% Compute score
score = [expValMdp - expValIrl; expValIrl; expValMdp];
