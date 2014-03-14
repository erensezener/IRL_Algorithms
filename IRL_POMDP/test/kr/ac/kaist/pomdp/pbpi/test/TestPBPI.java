package kr.ac.kaist.pomdp.pbpi.test;

import java.util.ArrayList;
import java.util.Random;

import no.uib.cipr.matrix.Vector;

import junit.framework.TestCase;
import kr.ac.kaist.pomdp.data.BeliefPoints;
import kr.ac.kaist.pomdp.data.FSC;
import kr.ac.kaist.pomdp.data.PomdpFile;
import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.pomdp.pbpi.PBPI;

public class TestPBPI extends TestCase {
	final public long seed = 0;
	
	/**
	 * junit test case for PBPI
	 * 
	 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
	 */
	public void testRun() throws Exception {		
		String probName = "Tiger"; 
//		String probName = "Maze_1d";
//		String probName = "Grid_5x5_z9";
//		String probName = "heaven-hell";
//		String probName = "RockSample_4x3";
//		String probName = "RockSample_4_4";
//		String probName = "milos-aaai97S";
//		String probName = "Maze_4x3";
		
		boolean useSparse = true;
		Random rand = new Random(seed);
		int maxRestart = 10;				// number of iterations in single expansion step
		int nBeliefs = 50;					// number of expansion of sampled belief points
		double minDistBeliefs = 1e-1;		// minimum distance between sampled belief points

		int maxIters = 1000;				// number of iterations of pbpi
		double minConvError = 1e-6;			// minimum error for convergence
		boolean bPrint = true; //false;

		double eps = 1e-4;
		int simMaxIters = 5000;				// number of episodes for simulation
		int simMaxSteps = 20;				// number of steps in single episode for simulation
		boolean simPrint = false;
				
		// Read POMDP file
		System.out.printf("Problem : %s\n\n", probName);
		String pomdpFileName = "./problems/" + probName + ".pomdp";
		String fscFileName = "./output/" + probName + ".pbpi";
		
		PomdpProblem pomdpProb = PomdpFile.read(pomdpFileName, useSparse);
//		pomdpProb.gamma = 0.75;
		pomdpProb.printBriefInfo();
		//pomdpProb.printInfo();
		//pomdpProb.printSparseInfo();

//		// Simulate the computed FSC
//		FSC fsc = new FSC(pomdpProb);
//		fsc.read(fscFileName);
//		fsc.print();
//		fsc.findReachableBeliefs();
		
		// Sample beliefs
		ArrayList<Vector> beliefSet = 
			BeliefPoints.initBeliefs(pomdpProb, nBeliefs, maxRestart, minDistBeliefs, rand);
		//Mtrx.print(beliefSet);
		System.out.printf("# of samples beliefs : %d\n\n", beliefSet.size());
		
		// Run PBPI
		PBPI pbpi = new PBPI(pomdpProb, beliefSet);
		pbpi.setParams(maxIters, minConvError);
		FSC fsc = pbpi.run(bPrint, rand);
		pbpi.print(fsc);
		pbpi.print(fsc, fscFileName);
			
		// Simulate
		double tmp = eps * (1.0 - pomdpProb.gamma) / pomdpProb.RMAX;
		simMaxSteps = (int) (Math.log(tmp) / Math.log(pomdpProb.gamma));
		fsc.simulate(simMaxIters, simMaxSteps, simPrint, rand);
		
		// print simulation
		//fsc.simulate(3, 50, true, rand);
		
		// Release
		fsc.delete();
		fsc = null;
		rand = null;
		pbpi.delete();
		pomdpProb.delete();
	}	
}
