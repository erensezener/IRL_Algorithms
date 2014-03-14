% Count state-action vistation in example samples
function smpl = countsamples(example_samples, nS, nA)

smpl.cnt = sparse(nS, nA);
for i = 1:size(example_samples, 1)
    for t = 1:size(example_samples, 2)
        s = example_samples{i,t}(1);
        a = example_samples{i,t}(2);
        smpl.cnt(s, a) = smpl.cnt(s, a) + 1;
    end
end
smpl.p = smpl.cnt + ones(nS, nA);
smpl.p = bsxfun(@rdivide, smpl.p, sum(smpl.p, 2));

[i, j, k]  = find(smpl.cnt);
smpl.spcnt = [i, j, k];

end
