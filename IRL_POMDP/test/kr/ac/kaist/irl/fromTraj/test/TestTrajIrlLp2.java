package kr.ac.kaist.irl.fromTraj.test;

import java.util.Random;

import junit.framework.TestCase;
import kr.ac.kaist.irl.fromTraj.TrajIrlLp2;
import kr.ac.kaist.pomdp.data.BasisFunctions;
import kr.ac.kaist.pomdp.data.PomdpFile;
import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.utils.CpuTime;
import kr.ac.kaist.utils.IrlUtil;

/**
 * junit test case for IRL using the max-margin between values method.
 * 
 * maximize t - lambda ||R||_1
 * subject to alpha*mu_E >= alpha*mu_pi + t for all pi
 *            |alpha_i| <= 1 for all i
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class TestTrajIrlLp2 extends TestCase {
	private long seed = 1;
	private double trajEps = 1e-2;
	private double goodRatio = 0.95;
	
	private String probName;
	private String pomdpFileName;
	private String fscFileName;
	private int nExps = 100;				// # of experiments
	private int nTrajs = 2000;				// # of trajectories
	private int nSteps = 20;				// # of steps in each trajectory
	private int nIrlIters = 30;				// # of IRL iterations
	private int nSampledBeliefs = 100;		// # of sampled beliefs for PBPI
	private double lambda = 0;				// weight of the penalty on the norm of reward functions
	private double minDist = 1e-5;			// minimum distance between beliefs
	private double irlEps = 1e-5;
	private double goodReturn;
	private String basisType = "SA"; 
	private boolean useSparse = true;
	private boolean bPrint = true;
	private Random rand;
		
	public void testRun() throws Exception {
//		probName = "Tiger";
		probName = "Maze_1d";
//		probName = "Grid_5x5_z9";
//		probName = "heaven-hell";
//		probName = "RockSample_4_4";
//		probName = "RockSample_4x3";
		basisType = "SA";
		
		PomdpProblem pomdpProb = configure();		
		System.out.printf("Test Trajectory LP-IRL (2)\n\n");	
		System.out.printf("+ Problem : %s (%s)\n\n", probName, basisType);
		pomdpProb.printBriefInfo();
		System.out.printf("+ Epsilon for trajectories      : %f\n", trajEps);
		System.out.printf("+ Epsilon for IRL               : %f\n", irlEps);		
		System.out.printf("+ Lambda for IRL                : %f\n", lambda);
		System.out.printf("+ # of experiments              : %d\n", nExps);
		System.out.printf("+ # of IRL iterations           : %d\n", nIrlIters);
		System.out.printf("+ # of trajectories             : %d\n", nTrajs);
		System.out.printf("+ # of steps                    : %d\n", nSteps);
		System.out.printf("+ # of sampled beliefs for PBPI : %d\n", nSampledBeliefs);
		System.out.printf("+ Min. distance between beliefs : %f\n", minDist);
		System.out.printf("+ Good return                   : %f\n\n", goodReturn);
		pomdpProb.delete();
		pomdpProb = null;		
		repeatedExp();
	}
	
	private PomdpProblem configure() throws Exception {		
		if (probName.equals("Tiger")) {
			irlEps = 0.005;
			trajEps = 0.01; //36
			nIrlIters = 10;
			//5, 10, 20, 50, 100, 200, 500, 1000, 2000
			nTrajs = 5; 
		}
		else if (probName.equals("Maze_1d")) {
//			irlEps = 0.005;
//			trajEps = 0.01; //20
//			nIrlIters = 10;
//			//5, 10, 20, 50, 100, 200, 500, 1000, 2000
//			nTrajs = 2000;
			
			irlEps = 0.005;
			trajEps = 1; //4
			nIrlIters = 10;
			//5, 10, 20, 50, 100, 200, 500, 1000, 2000
			nTrajs = 2000;
		}
		else if (probName.equals("Grid_5x5_z9")) {
//			lambda = 0.1;		
//			trajEps = 0.01; //65
//			nIrlIters = 15;
			
			lambda = 0.1;	
			trajEps = 1.5; //65
			nIrlIters = 15;
			//5, 10, 20, 50, 100, 200, 500, 1000, 2000
			nTrajs = 500;
			nExps = 10;
		}
		else if (probName.equals("heaven-hell")) {
			if (basisType.equals("SA")) {
				irlEps = 0.005;
				lambda = 0.5;
				nIrlIters = 20;
				nExps = 20;
			}
			else if (basisType.equals("S")) {
				irlEps = 0.01; //0.02;
				lambda = 0.5;
				nIrlIters = 15;
				nExps = 20;
			}
			else if (basisType.equals("C")) {
//				trajEps = 0.01; //916
//				nIrlIters = 15;
				
				trajEps = 1.5; //458
				nIrlIters = 15;
				//5, 10, 20, 50, 100, 200, 500, 1000, 2000
				nTrajs = 500;
				nExps = 50;
			}
		}
		else if (probName.equals("RockSample_4x3")) {
			nExps = 10;
			nSampledBeliefs = 100;
			minDist = 1e-1;
			if (basisType.equals("SA")) {
				irlEps = 0.06;
//				lambda = 1.0;
				nIrlIters = 50;
			}
			else if (basisType.equals("NC")) {
				irlEps = 0.03;
//				lambda = 0.5;
				nIrlIters = 50;
			}
			else if (basisType.equals("C")) {
				irlEps = 0.001;
				nIrlIters = 50;
			}
		}
		else if (probName.equals("RockSample_4_4")) {
			if (basisType.equals("SA")) {
				irlEps = 1e-5; //1.0;
				lambda = 1.0;
				nExps = 10;
				nIrlIters = 50;
			}
			else if (basisType.equals("NC")) {
				irlEps = 0.03; //0.01; //0.03; //0.1;
				lambda = 0.5;
				nExps = 10;
				nIrlIters = 50;
			}
			else if (basisType.equals("C")) {
				irlEps = 0.03;
				nExps = 20;
				nIrlIters = 20;
			}
			nSampledBeliefs = 100;
			minDist = 1e-1;
		}

		pomdpFileName = "./problems/" + probName + ".pomdp";
		fscFileName = "./output/" + probName + ".pbpi";
		PomdpProblem tmpPomdpProb = PomdpFile.read(pomdpFileName, useSparse);

		goodReturn = IrlUtil.getGoodReturn(probName, useSparse);
		double x = trajEps * (1.0 - tmpPomdpProb.gamma) / tmpPomdpProb.RMAX;
		nSteps = (int) (Math.log(x) / Math.log(tmpPomdpProb.gamma));
		rand = new Random(seed);
		return tmpPomdpProb;
	}
		
	public void repeatedExp() throws Exception {
		double expDiffV0 = 0;
		double varDiffV0 = 0;
		double[][] expEps = new double[nExps][nIrlIters];
		double[][] expDiffV = new double[nExps][nIrlIters];
		double[][] expTrueV = new double[nExps][nIrlIters];
		double[][] expDiffR = new double[nExps][nIrlIters];
		int[] cnt = new int[nIrlIters];
		for (int i = 0; i < nIrlIters; i++) {
			cnt[i] = 0;
			for (int j = 0; j < nExps; j++) {
				expEps[j][i] = Double.NaN;
				expDiffV[j][i] = Double.NaN;
				expTrueV[j][i] = Double.NaN;
				expDiffR[j][i] = Double.NaN;
			}
		}
		int expIter = 0; 
		int goodReturnCnt = 0;
		double irlTime = 0;
		double pbpiTime = 0;
		double meanReturn = 0;
		double varReturn = 0;
		double startTime = CpuTime.getCurTime();
		
		for (int t = 0; t < nExps; t++) {
			System.out.printf("=== %d-th iteration ==============================================\n", t);
			PomdpProblem pomdpProb = PomdpFile.read(pomdpFileName, useSparse);
			//pomdpProb.printBriefInfo();
			
			BasisFunctions phi = new BasisFunctions(pomdpProb, probName, basisType);			
			TrajIrlLp2 irl = new TrajIrlLp2(pomdpProb);
			irl.setParams(fscFileName, phi, nIrlIters, 
					nSampledBeliefs, irlEps, 
					nTrajs, nSteps, minDist, rand);
			irl.solve(lambda, bPrint);
			
			int minId = -1;
			double minDiff = Double.POSITIVE_INFINITY;
			for (int i = 0; i < irl.diffVList.size(); i++) {
				double x = irl.diffVList.get(i);
				if (minDiff > x && !irl.checkReward(i)) {
					minDiff = x;
					minId = i;
				}
			}
			//int minId = trajIrl.trueVList.size() - 1;
			for (int i = 0; i <= minId; i++) {
				expEps[t][i] = irl.epsList.get(i);
				expDiffV[t][i] = irl.diffVList.get(i);
				expTrueV[t][i] = irl.trueVList.get(i);
				expDiffR[t][i] = irl.diffRList.get(i);
				cnt[i]++;
			}			
			for (int i = minId + 1; i < nIrlIters; i++) {
				expEps[t][i] = expEps[t][minId];
				expDiffV[t][i] = expDiffV[t][minId];
				expTrueV[t][i] = expTrueV[t][minId];
				expDiffR[t][i] = expDiffR[t][minId];
				cnt[i]++;				
			}
			if (expTrueV[t][minId] >= goodRatio * goodReturn) goodReturnCnt++;
			expIter += minId;
			meanReturn += expTrueV[t][minId];
			varReturn += expTrueV[t][minId] * expTrueV[t][minId];
			irlTime += irl.totalIrlTime;
			pbpiTime += irl.totalPbpiTime;	
			expDiffV0 += irl.diffV;
			varDiffV0 += irl.diffV * irl.diffV;		

			System.out.printf("- Eps         : %f\n", expEps[t][minId]);
			System.out.printf("- Diff(V;R')  : %f\n", expDiffV[t][minId]);
			System.out.printf("- V^pi(R)     : %f\n", expTrueV[t][minId]);
			System.out.printf("- |R-R'|_w,1  : %f\n\n", expDiffR[t][minId]);
			
			// release 
			irl.delete();
			irl = null;
			pomdpProb.delete();
			pomdpProb = null;
		}
		expDiffV0 /= nExps;
		varDiffV0 = varDiffV0 / nExps - expDiffV0 * expDiffV0;
		meanReturn /= nExps;
		varReturn = varReturn / nExps - meanReturn * meanReturn;
		double totalTime = CpuTime.getElapsedTime(startTime); 

		System.out.println();
		System.out.println("=================================================================");
		System.out.println("Test Trajectory LP-IRL (2)");	
		System.out.printf("+ Problem                       : %s (%s)\n", probName, basisType);
		System.out.printf("+ Epsilon for IRL               : %f\n", irlEps);		
		System.out.printf("+ Lambda for IRL                : %f\n", lambda);
		System.out.printf("+ # of experiments              : %d\n", nExps);
		System.out.printf("+ # of IRL iterations           : %d\n", nIrlIters);
		System.out.printf("+ # of trajectories             : %d\n", nTrajs);
		System.out.printf("+ # of steps                    : %d (%f)\n", nSteps, trajEps);
		System.out.printf("+ # of sampled beliefs for PBPI : %d\n", nSampledBeliefs);
		System.out.printf("+ Good return                   : %f\n\n", goodReturn);
		System.out.println("=== E[V(t)] =====================================================");
		System.out.println(" Run |   E[DiffV]   S[DiffV]   E[TrueV]   S[TrueV]   " +
				"E[DiffR]   S[DiffR]  cnt");
		for (int t = 0; t < nIrlIters; t++) {
			if (cnt[t] > 0) {
				double meanEps = 0;
				double meanDiffV = 0;
				double meanTrueV = 0;
				double meanDiffR = 0;
				for (int i = 0; i < nExps; i++) { 
					if (!Double.isNaN(expEps[i][t])) meanEps += expEps[i][t];
					if (!Double.isNaN(expDiffV[i][t])) meanDiffV += expDiffV[i][t];
					if (!Double.isNaN(expTrueV[i][t])) meanTrueV += expTrueV[i][t];
					if (!Double.isNaN(expDiffR[i][t])) meanDiffR += expDiffR[i][t];
				}
				meanEps /= cnt[t];
				meanDiffV /= cnt[t];
				meanTrueV /= cnt[t];
				meanDiffR /= cnt[t];
				
				double varEps = 0;
				double varDiffV = 0;
				double varTrueV = 0;
				double varDiffR = 0;
				if (cnt[t] > 1) {
					for (int i = 0; i < nExps; i++) {
						if (!Double.isNaN(expEps[i][t])) varEps += Math.pow(expEps[i][t] - meanEps, 2);
						if (!Double.isNaN(expDiffV[i][t])) varDiffV += Math.pow(expDiffV[i][t] - meanDiffV, 2);
						if (!Double.isNaN(expTrueV[i][t])) varTrueV += Math.pow(expTrueV[i][t] - meanTrueV, 2);
						if (!Double.isNaN(expDiffR[i][t])) varDiffR += Math.pow(expDiffR[i][t] - meanDiffR, 2);
					}
					varEps /= (cnt[t] - 1);
					varDiffV /= (cnt[t] - 1);
					varTrueV /= (cnt[t] - 1);
					varDiffR /= (cnt[t] - 1);
				}
				
				System.out.printf(" %3d | %10.4f %10.4f %10.4f %10.4f" +
						" %10.4f %10.4f %4d\n", 
						t, meanDiffV, varDiffV, meanTrueV, varTrueV, 
						meanDiffR, varDiffR, cnt[t]);
			}
		}		
		System.out.println("=================================================================");
		System.out.println();		
		System.out.printf("+ Avg. Return                   : %f %f\n", meanReturn, varReturn);
		System.out.printf("+ |V*-V_E|                      : %f %f\n", expDiffV0, varDiffV0);
		System.out.printf("+ # of Good Return              : %d / %d (%.2f)\n", 
				goodReturnCnt, nExps, (double) goodReturnCnt / nExps * 100);
		System.out.printf("+ Avg. # of Iterations          : %.4f\n", (double) expIter / nExps);
		System.out.printf("+ Elapsed time                  : %.4f sec\n", totalTime);	
		System.out.printf("+ Irl Time                      : %.4f sec\n", irlTime);
		System.out.printf("+ Pbpi Time                     : %.4f sec (%.2f)\n\n", 
				pbpiTime, (pbpiTime / irlTime * 100));
	}
}
