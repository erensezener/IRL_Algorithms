% Generate sparse state-transition matrix multiplied by discount factor
% Written by Jaedeug Choi
function TG = gendiscountedtransmtrx(mdp_data)

[nS, nA, ~] = size(mdp_data.sa_p);
TG = cell(nA, 1);
for a = 1:nA
    TG{a} = sparse(nS, nS);
    for s = 1:nS
        for k = 1:length(mdp_data.sa_s(s, a, :))
            s2           = mdp_data.sa_s(s, a, k);
            TG{a}(s, s2) = TG{a}(s, s2) + mdp_data.sa_p(s, a, k);
        end
    end
    TG{a} = TG{a}.*mdp_data.discount;
end

end