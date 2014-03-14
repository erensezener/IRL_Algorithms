% Remove unnecessary paths for subdirectories.

% External dependencies.
rmpath Utilities
rmpath Utilities/minFunc

% General functionality.
rmpath General
rmpath Evaluation

% MDP solvers.
rmpath StandardMDP
rmpath LinearMDP

% IRL algorithms.
rmpath FIRL
rmpath GPIRL
rmpath MaxEnt
rmpath MMP
rmpath BirlGA
rmpath NPBFIRL

% Example MDPs.
rmpath Gridworld
rmpath Objectworld
rmpath Highway

% cvx
rmpath Utilities/cvx
rmpath Utilities/cvx/structures
rmpath Utilities/cvx/lib
rmpath Utilities/cvx/functions
rmpath Utilities/cvx/commands
rmpath Utilities/cvx/builtins

