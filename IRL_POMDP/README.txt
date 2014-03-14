Inverse reinforcement learning in partially observable environments

- Jaedeug Choi (jdchoi@ai.kaist.ac.kr)

This package solves inverse reinforcement learning (IRL) problems when the environments are modeled as partially observable Markov decision processes (POMDPs). The algorithms are described in [ChoiKim.09; ChoiKim.11].

# Requirement
This package uses IBM ILOG CPLEX. Thus, CPLEX should be installed in your machine in order to execute this package. Additionally, you should set the java library path to CPLEX before executing this package.

# Package overview
- kr.ac.kaist.irl.fromFSC
  Solve IRL problems for POMDPs when the expert policy is explicitly given in the form of a finite state controller (FSC).
- kr.ac.kaist.irl.fromTraj
  Solve IRL problems for POMDPs when the behavior data is given by trajectories of the expert¡¯s executed actions and the corresponding observations.
- kr.ac.kaist.pomdp.pbpi
  Solve POMDPs using point-based policy iteration [JiETAL.07] 

# Usage: use junit test cases.

[ChoiKim.09] J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, IJCAI 2009.
[ChoiKim.11] J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, JMLR 12, 2011.
[JiETAL.07] S. Ji, R. Parr, H. Li et al., Point-based policy iteration, AAAI 2007.



