function setrandomseed(seed)

RandStream.setGlobalStream(RandStream.create('mrg32k3a', 'seed', seed));

% if verLessThan('matlab', '8.0.0')
%     % MATLAB 8.0.0 and earlier
%     RandStream.setDefaultStream(RandStream.create('mrg32k3a', 'seed', seed));
% else
%     % MATLAB 8.0.1 and later
%     rng(seed);
% end
% pause(0.1);

end