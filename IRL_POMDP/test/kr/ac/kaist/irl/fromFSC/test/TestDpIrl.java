package kr.ac.kaist.irl.fromFSC.test;

import java.util.Random;

import junit.framework.TestCase;
import kr.ac.kaist.irl.fromFSC.DpIrl;
import kr.ac.kaist.pomdp.data.FSC;
import kr.ac.kaist.pomdp.data.PomdpFile;
import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.utils.CpuTime;

/**
 * junit test case for IRL using the dynamic programming (DP) update based optimality constraint.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class TestDpIrl extends TestCase {
	final private long seed = 0;
	final private boolean useConfigure = false;
		
	private String probName;
	private double lBofRewardNorm = 1;		// lower bound of the norm of reward functions
	private double uBofRewardNorm = 2;		// upper bound of the norm of reward functions
	private double lambda = 1e-8;			// weight of the penalty on the norm of reward functions
	private int maxIter = 50;				// # of IRL iterations
	private int nBeliefs = 100;				// # of sampled beliefs for PBPI
	private int beliefSamplingT = 500;		// belief sampling time
	private int beliefSamplingH = 200;		// belief sampling horizon
	private double minDist = 1e-5;			// minimum distance between beliefs
	private boolean irlPrint = false;
	private boolean expPrint = false;
	private Random rand;
	
	public void testRun() throws Exception {			
//		probName = "Tiger"; 
//		probName = "Maze_1d";
//		probName = "Grid_5x5_z9";
//		probName = "heaven-hell";
//		probName = "RockSample_4_4";
		probName = "RockSample_4x3";

		rand = new Random(seed);	
//		singleExp();
//		repeatedExp();
		
		int N = 10;
		double totalTime = 0;
		for (int i = 0; i < N; i++) {
			System.out.println("*******************************************************");
			totalTime += singleExp();
			System.out.println();
		}
		System.out.println();
		System.out.println("*******************************************************");
		System.out.printf("Avg. IRL time : %f sec\n", totalTime / N);
		System.out.println("*******************************************************");
		System.out.println();
	}
	
	public void configureSingleExp() {		
		if (probName.equals("Tiger")) {
			lambda = 85.679758;
		}
		else if (probName.equals("Maze_1d")) {
			lambda = 25;
		}
		else if (probName.equals("Grid_5x5_z9")) {
			lambda = 5116;
		}
	}
	
	public void configureRepeatedExp() {		
		if (probName.equals("Tiger")) {
			lBofRewardNorm = 2;
			uBofRewardNorm = 2.2;
		}
		else if (probName.equals("Maze_1d")) {
			lBofRewardNorm = 0.5;
			uBofRewardNorm = 1.5;
		}
		else if (probName.equals("Grid_5x5_z9")) {
			lBofRewardNorm = 3;
			uBofRewardNorm = 4;	
			maxIter = 100;		
		}
		else if (probName.equals("RockSample_4_4")) {
			lBofRewardNorm = 16;
			uBofRewardNorm = 10;
			maxIter = 100;
			nBeliefs = 50;
			minDist = 1e-1;
		}
	}
	
	public double singleExp() throws Exception {
		if (useConfigure) configureSingleExp();

		boolean useSparse = true;
		System.out.printf("DP-IRL : %s\n\n", probName);
		System.out.printf("+ # of sampled beliefs for PBPI : %d\n", nBeliefs);
		System.out.printf("+ Belief sampling time          : %d\n", beliefSamplingT);
		System.out.printf("+ Belief sampling horizon       : %d\n\n", beliefSamplingH);
		
		String pomdpFileName = "./problems/" + probName + ".pomdp";
		String fscFileName = "./output/" + probName + ".pbpi";
		//String fscFileName = "./output/" + probName + ".witness";
		
		PomdpProblem pomdpProb = PomdpFile.read(pomdpFileName, useSparse);
		pomdpProb.printBriefInfo();
		//pomdpProb.printSparseInfo();
		
		// read fsc and sample beliefs
		FSC fsc = new FSC(pomdpProb);
		fsc.read(fscFileName);
		fsc.sampleNodeBelief(beliefSamplingT, beliefSamplingH, rand);

		fsc.print();		
		double optV = fsc.getV0();
		System.out.printf("Nodes of the Opt. FSC : %d\n", fsc.size());
		System.out.printf("Value of the Opt. FSC : %f\n\n", optV);
		
		// inverse reinforcement learning based on policy improvement thm.
		DpIrl irl = new DpIrl(pomdpProb, fsc, nBeliefs, minDist, irlPrint, rand);
		System.out.println("=== Dp-Irl ============================================");
		
		double T0 = CpuTime.getCurTime();
		double obj = irl.solve(lambda);
		double endTime = CpuTime.getElapsedTime(T0);

		System.out.printf("Elapsed Time : %f sec\n", endTime);			
		System.out.printf("nRows        : %d\n", irl.nRows);
		System.out.printf("nCols        : %d\n", irl.nCols);
		System.out.printf("Lambda       : %f\n", lambda);
		System.out.printf("Obj. Value   : %f\n\n", obj);
		
//		double[][] learnedR = irl.getReward();
//		double rewardNorm = irl.getSumR(learnedR);
//		double[] result = irl.eval(learnedR, rand);
//		double trueV = result[0];
//		double V_E = result[1];
//		double V_pi = result[2];
//		double norm = irl.calWeightedNorm(learnedR, 1);				
//		System.out.println("=== Learned reward ====================================");
//		System.out.printf("|R|_1        : %f\n", rewardNorm);
//		System.out.printf("V_pi(R)      : %f\n", trueV);
//		System.out.printf("Diff(R)      : %f\n", optV - trueV);
//		System.out.printf("Diff(R')     : %f\n", V_E - V_pi);
//		System.out.printf("|R-R'|_w,1   : %f\n", norm);
//		irl.printReward(learnedR);
		
//		double[][] transformedR = irl.transformReward(learnedR);
//		double rewardNorm2 = irl.getSumR(transformedR);
//		double[] result2 = irl.eval(transformedR, rand);
//		double trueV2 = result2[0];
//		double V_E2 = result2[1];
//		double V_pi2 = result2[2];
//		double norm2 = irl.calWeightedNorm(transformedR, 1);
//		System.out.println("=== Transformed reward ================================");
//		System.out.printf("|R|_1        : %f\n", rewardNorm2);
//		System.out.printf("V_pi(R)      : %f\n", trueV2);
//		System.out.printf("Diff(R)      : %f\n", optV - trueV2);
//		System.out.printf("Diff(R')     : %f\n", V_E2 - V_pi2);
//		System.out.printf("|R-R'|_w,1   : %f\n", norm2);
//		irl.printReward(transformedR);
				
		// release 
		irl.delete();
		irl = null;
		fsc.delete();
		fsc = null;
		pomdpProb.delete();
		pomdpProb = null;
		return endTime;
	}
	
	public void repeatedExp() throws Exception {
		if (useConfigure) configureSingleExp();

		boolean useSparse = true;
		System.out.printf("DP-IRL : %s\n\n", probName);
		System.out.printf("+ Lower bound of norm of reward : %.2f\n", lBofRewardNorm);
		System.out.printf("+ Upper bound of norm of reward : %.2f\n", uBofRewardNorm);
		System.out.printf("+ # of IRL iterations           : %d\n", maxIter);
		System.out.printf("+ # of sampled beliefs for PBPI : %d\n", nBeliefs);
		System.out.printf("+ Belief sampling time          : %d\n", beliefSamplingT);
		System.out.printf("+ Belief sampling horizon       : %d\n\n", beliefSamplingH);
		
		String pomdpFileName = "./problems/" + probName + ".pomdp";
		String fscFileName = "./output/" + probName + ".pbpi";
		//String fscFileName = "./output/" + probName + ".witness";
		
		PomdpProblem pomdpProb = PomdpFile.read(pomdpFileName, useSparse);
		pomdpProb.printBriefInfo();
		//pomdpProb.printSparseInfo();
		
		// read fsc and sample beliefs
		FSC fsc = new FSC(pomdpProb);
		fsc.read(fscFileName);
		fsc.sampleNodeBelief(beliefSamplingT, beliefSamplingH, rand);

		fsc.print();		
		double optV = fsc.getV0();
		System.out.printf("Nodes of the Opt. FSC : %d\n", fsc.size());
		System.out.printf("Value of the Opt. FSC : %f\n\n", optV);
		
		double lam = lambda;
		double incStep = 300;
				
		double[][] finalR = null;
		double[][] finalR2 = null;
		double finalObj = 0.0;
		double finalLam = 0.0;
		double finalTrueV = 0.0;
		
		double lowerB = Double.NEGATIVE_INFINITY;
		double upperB = Double.POSITIVE_INFINITY;
		double totalIrlTime = 0;
		double totalPbpiTime = 0;
		int nIter = 0;

		// inverse reinforcement learning based on policy improvement thm.	
		DpIrl irl = new DpIrl(pomdpProb, fsc, nBeliefs, minDist, irlPrint, new Random());
		System.out.println("=== Dp-Irl ============================================");
		if (!expPrint) 
			System.out.println(" Run |     Lambda      Obj. V      |R|      Diff(R)   Diff(R')     |R-R'|    |R-R2'| ::     Time");
		
		for (nIter = 0; nIter < maxIter && incStep > 1e-5; nIter++) {
			double T0 = CpuTime.getCurTime();
			double obj = irl.solve(lam);
			double endIrlTime = CpuTime.getElapsedTime(T0);
			totalIrlTime += endIrlTime;
			
			double[][] learnedR = irl.getReward();
			double rewardNorm = irl.getSumR(learnedR);
			T0 = CpuTime.getCurTime();
			double[] result = irl.eval(learnedR, rand);
			double pbpiEndTime = CpuTime.getElapsedTime(T0);
			totalPbpiTime += pbpiEndTime;
			double trueV = result[0];
			double V_E = result[1];
			double V_pi = result[2];
			double norm = irl.calWeightedNorm(learnedR, 1);	
			double diffV = V_E - V_pi;
			
			double[][] transformedR = irl.transformReward(learnedR);
			double norm2 = irl.calWeightedNorm(transformedR, 1);
			
			if (expPrint) {
				System.out.printf("=== %2d run =======================================\n", nIter);
				System.out.printf("nRows      : %d\n", irl.nRows);
				System.out.printf("nCols      : %d\n", irl.nCols);
				System.out.printf("Lambda     : %f\n", lam);
				System.out.printf("Obj. Value : %f\n", obj);
				System.out.printf("|R|        : %f\n", rewardNorm);
				System.out.printf("Diff. of V : %f\n", diffV);
				System.out.printf("True V     : %f / %f (%.2f)\n\n", 
						trueV, optV, trueV / optV * 100);
				irl.printReward(learnedR);
			}
			else
				System.out.printf(" %3d | %10.4f  %12.4f %8.2f %10.4f %10.4f %10.4f %10.4f " +
						":: %8.2f sec  r%d c%d n%d\n", 
						nIter, lam, obj, rewardNorm, optV - trueV, V_E - V_pi, norm, norm2, 
						endIrlTime, irl.nRows, irl.nCols, irl.fscSize);
			
			if (rewardNorm >= lBofRewardNorm) {
				finalR = learnedR;
				finalR2 = transformedR;
				finalObj = obj;
				finalLam = lam;
				finalTrueV = trueV;
			}
			
			if (rewardNorm > lBofRewardNorm && rewardNorm < uBofRewardNorm) break;
			else if (rewardNorm >= uBofRewardNorm) {
				lowerB = lam;
				if (lowerB >= upperB) upperB = lowerB;				
//				incStep *= 1.5; //2;
				lam += incStep;				
				if (lam > upperB) {
					incStep = (upperB - lowerB) / 2;
					lam = (upperB + lowerB) / 2;
				}
			}
			else /*if (rewardNorm <= lBofRewardNorm && rewardNorm > 0)*/ {
				upperB = lam;				
				incStep /= 1.5; //2;
				lam -= incStep;				
				if (lam < lowerB) {
					incStep = (upperB - lowerB) / 2;
					lam = (upperB + lowerB) / 2;
				}
			}
		}
		System.out.println("=======================================================\n");
		
		System.out.printf("Lambda     : %f\n", finalLam);
		System.out.printf("Obj. Value : %f\n", finalObj);
		System.out.printf("True V     : %f / %f (%.2f)\n\n", 
				finalTrueV, optV, finalTrueV / optV * 100);
		
		for (int s = 0; s < pomdpProb.nStates; s++) {
			for (int a = 0; a < pomdpProb.nActions; a++) {
				if (finalR[s][a] != 0) {
					if (pomdpProb.states == null) System.out.printf("R[s%d]", s);
					else System.out.printf("R[%s]", pomdpProb.states[s]);
					if (pomdpProb.actions == null) System.out.printf("[a%d] : %.20f\n", a, finalR[s][a]);
					else System.out.printf("[%s] : %.20f\n", pomdpProb.actions[a], finalR[s][a]);
				}
			}
		}
		System.out.println();
		
//		for (int s = 0; s < pomdpProb.nStates; s++) {
//			for (int a = 0; a < pomdpProb.nActions; a++) {
//				if (finalR2[s][a] != 0) {
//					if (pomdpProb.states == null) System.out.printf("R[s%d]", s);
//					else System.out.printf("R[%s]", pomdpProb.states[s]);
//					if (pomdpProb.actions == null) System.out.printf("[a%d] : %.20f\n", a, finalR2[s][a]);
//					else System.out.printf("[%s] : %.20f\n", pomdpProb.actions[a], finalR2[s][a]);
//				}
//			}
//		}
		System.out.println("=======================================================");
		System.out.printf("# of Iterations    : %d\n", nIter++);
		System.out.printf("Avg. IRL time      : %f sec\n", totalIrlTime / nIter);
		System.out.printf("Avg. PBPI time     : %f sec\n", totalPbpiTime / nIter);
		System.out.printf("Total IRL time     : %f sec\n", totalIrlTime);
		System.out.printf("Total PBPI time    : %f sec\n", totalPbpiTime);
				
		// release 
		irl.delete();
		irl = null;
		fsc.delete();
		fsc = null;
		pomdpProb.delete();
		pomdpProb = null;
	}	
}
