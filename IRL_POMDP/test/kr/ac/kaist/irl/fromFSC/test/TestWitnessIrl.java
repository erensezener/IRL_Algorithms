package kr.ac.kaist.irl.fromFSC.test;

import java.util.Random;

import junit.framework.TestCase;
import kr.ac.kaist.irl.fromFSC.WitnessIrl;
import kr.ac.kaist.pomdp.data.BeliefPoints;
import kr.ac.kaist.pomdp.data.FSC;
import kr.ac.kaist.pomdp.data.PomdpFile;
import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.utils.CpuTime;

/**
 * junit test case for IRL using the witness theorem based optimality constraint.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class TestWitnessIrl extends TestCase {
	private long seed = 0;
	final private boolean useConfigure = true;
	final private double initialInc = 50;
	
	private String probName;
	private String pomdpFileName;
	private String fscFileName;
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
		singleExp();
//		repeatedExp();
		
//		int N = 10;
//		double totalTime = 0;
//		for (int i = 0; i < N; i++) {
//			System.out.println("*******************************************************");
//			totalTime += singleExp();
//			System.out.println();
//		}
//		System.out.println();
//		System.out.println("*******************************************************");
//		System.out.printf("Avg. IRL time : %f sec\n", totalTime / N);
//		System.out.println("*******************************************************");
//		System.out.println();
	}
		
	public PomdpProblem configureSingleExp() throws Exception {
		pomdpFileName = "./problems/" + probName + ".pomdp";
		fscFileName = "./output/" + probName + ".pbpi";
		
		if (useConfigure) {
			if (probName.equals("Tiger")) {
				lambda = 31.048918;
				//lambda = 25; numerical error is induced
			}
			else if (probName.equals("Maze_1d")) {
				lambda = 12.5;
			}
			else if (probName.equals("Grid_5x5_z9")) {
				lambda = 2887; //4465;
			}
			else if (probName.equals("heaven-hell")) {
				lambda = 164826;
			}
			else if (probName.equals("RockSample_4_4")) {
				seed = 1234; //123; //12;
				lambda = 1450;
				nBeliefs = 50;
				minDist = 1e-1;
			}
			else if (probName.equals("RockSample_4x3")) {
				lambda = 2300; //400;
				nBeliefs = 200;
				minDist = 1e-1;
			}
		}
		boolean useSparse = true;
		return PomdpFile.read(pomdpFileName, useSparse);
	}
	
	public PomdpProblem configureRepeatedExp() throws Exception {
		pomdpFileName = "./problems/" + probName + ".pomdp";
		fscFileName = "./output/" + probName + ".pbpi";
//		fscFileName = "./output/" + probName + ".witness";
				
		if (useConfigure) {
			if (probName.equals("Tiger")) {
				lBofRewardNorm = 2;
				uBofRewardNorm = 2.2;
			}
			else if (probName.equals("Maze_1d")) {
				lBofRewardNorm = 0.5;
				uBofRewardNorm = 1.5;
			}
			else if (probName.equals("Grid_5x5_z9")) {
				//lambda = 2800;
				lBofRewardNorm = 3;
				uBofRewardNorm = 4;
			}
			else if (probName.equals("heaven-hell")) {
				lBofRewardNorm = 11;
				uBofRewardNorm = 13;
				maxIter = 100;
			}
			else if (probName.equals("RockSample_4x3")) {
				lBofRewardNorm = 1;
				uBofRewardNorm = 10;
				maxIter = 100;
				nBeliefs = 50;
				minDist = 1e-1;
			}
			else if (probName.equals("RockSample_4_4")) {
				lBofRewardNorm = 16;
				uBofRewardNorm = 10;
				maxIter = 100;
				nBeliefs = 50;
				minDist = 1e-1;
			}
		}
		boolean useSparse = true;
		return PomdpFile.read(pomdpFileName, useSparse);
	}
	
	public double singleExp() throws Exception {
		PomdpProblem pomdpProb = configureSingleExp();

		System.out.printf("Witness-IRL : %s\n\n", probName);
		System.out.printf("+ # of sampled beliefs for PBPI : %d\n", nBeliefs);
		System.out.printf("+ Belief sampling time          : %d\n", beliefSamplingT);
		System.out.printf("+ Belief sampling horizon       : %d\n\n", beliefSamplingH);
				
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
		
		// inverse reinforcement learning based on witness thm.
		WitnessIrl irl = new WitnessIrl(pomdpProb, fsc, nBeliefs, minDist, irlPrint, rand);
		System.out.println("=== Witness-Irl =======================================");
		
		double T0 = CpuTime.getCurTime();
		double obj = irl.solve(lambda);
		double endTime = CpuTime.getElapsedTime(T0);

		System.out.printf("Elapsed Time : %f sec\n", endTime);			
		System.out.printf("nRows        : %d\n", irl.nRows);
		System.out.printf("nCols        : %d\n", irl.nCols);
		System.out.printf("Lambda       : %f\n", lambda);
		System.out.printf("Obj. Value   : %f\n\n", obj);
		
		double[][] learnedR = irl.getReward();
		double rewardNorm = irl.getSumR(learnedR);
		double[] result = irl.eval(learnedR, rand);
		double trueV = result[0];
		double V_E = result[1];
		double V_pi = result[2];
		double norm = irl.calWeightedNorm(learnedR, 1);				
		System.out.println("=== Learned reward ====================================");
		System.out.printf("|R'|_1       : %f\n", rewardNorm);
		System.out.printf("V_pi(R)      : %f\n", trueV);
		System.out.printf("Diff(R)      : %f\n", optV - trueV);
		System.out.printf("Diff(R')     : %f\n", V_E - V_pi);
		System.out.printf("|R-R'|_w,1   : %f\n", norm);
//		irl.printReward(learnedR);
						
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
		PomdpProblem pomdpProb = configureRepeatedExp();

		System.out.printf("Witness-IRL : %s\n\n", probName);
		System.out.printf("+ Lower bound of norm of reward : %.2f\n", lBofRewardNorm);
		System.out.printf("+ Upper bound of norm of reward : %.2f\n", uBofRewardNorm);
		System.out.printf("+ # of IRL iterations           : %d\n", maxIter);
		System.out.printf("+ # of sampled beliefs for PBPI : %d\n", nBeliefs);
		System.out.printf("+ Belief sampling time          : %d\n", beliefSamplingT);
		System.out.printf("+ Belief sampling horizon       : %d\n\n", beliefSamplingH);
				
		pomdpProb.printBriefInfo();
		//pomdpProb.printSparseInfo();
		
		// read fsc and sample beliefs
		FSC fsc = new FSC(pomdpProb);
		fsc.read(fscFileName);
		fsc.sampleNodeBelief(beliefSamplingT, beliefSamplingH, rand);
//		int nAdditionalBeliefs = 80;
//		fsc.sampleNodeBelief(BeliefPoints.initBeliefs(pomdpProb, nAdditionalBeliefs, 100, minDist, rand));

		fsc.print();		
		double optV = fsc.getV0();
		System.out.printf("Nodes of the Opt. FSC : %d\n", fsc.size());
		System.out.printf("Value of the Opt. FSC : %f\n\n", optV);
		
		double lam = lambda;
		double incStep = initialInc;
				
		double[][] finalR = null;
		double finalObj = 0;
		double finalLam = 0;
		double finalTrueV = 0;
		
		double lowerB = Double.NEGATIVE_INFINITY;
		double upperB = Double.POSITIVE_INFINITY;
		double totalIrlTime = 0;
		double totalPbpiTime = 0;
		int nIter = 0;

		// inverse reinforcement learning based on witness thm.
		WitnessIrl irl = new WitnessIrl(pomdpProb, fsc, nBeliefs, minDist, irlPrint, rand);
		System.out.println("=== Witness-Irl =======================================");
		if (!expPrint) 
			System.out.println(" Run |     Lambda      Obj. V      |R|      Diff(R)   Diff(R')     |R-R'| :: Time");
		
		for (nIter = 0; nIter < maxIter && incStep > 1e-5; nIter++) {
			double T0 = CpuTime.getCurTime();
			double obj = irl.solve(lam);
			double irlEndTime = CpuTime.getElapsedTime(T0);
			totalIrlTime += irlEndTime;
			
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
			else if (rewardNorm > 0)
				System.out.printf(" %3d | %10.4f  %12.4f %8.2f %10.4f %10.4f %10.4f " +
						":: %5.2f %5.2f  r%d c%d n%d\n", 
						nIter, lam, obj, rewardNorm, optV - trueV, V_E - V_pi, norm, 
						irlEndTime, pbpiEndTime,
						irl.nRows, irl.nCols, irl.fscSize);
			else
				System.out.println("   .");
			
			if (rewardNorm >= lBofRewardNorm) {
				finalR = learnedR;
				finalObj = obj;
				finalLam = lam;
				finalTrueV = trueV;
			}
			
			if (rewardNorm > lBofRewardNorm && rewardNorm < uBofRewardNorm) break;
			else if (rewardNorm >= uBofRewardNorm) {
				lowerB = lam;
				if (lowerB >= upperB) upperB = lowerB;				
//				incStep *= 2;
				lam += incStep;				
				if (lam > upperB) {
					incStep = (upperB - lowerB) / 2;
					lam = (upperB + lowerB) / 2;
				}
			}
			else /*if (rewardNorm <= lBofRewardNorm && rewardNorm > 0)*/ {
				upperB = lam;				
				incStep /= 2;
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
