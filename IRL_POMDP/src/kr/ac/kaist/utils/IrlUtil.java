package kr.ac.kaist.utils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;

import kr.ac.kaist.pomdp.data.*;
import kr.ac.kaist.pomdp.pbpi.PBPI;

/**
 * Some utilities 
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class IrlUtil {
	
	public static Vector initialWeight(int phiNum, double R_MIN, double R_MAX, 
			boolean useSparse, Random rand) {
		Vector alpha = Mtrx.Vec(phiNum, useSparse);
		for (int i = 0; i < phiNum; i++) {
			double r = rand.nextDouble() * (R_MAX - R_MIN) + R_MIN;
			if (r != 0) alpha.set(i, r);
		}

//		while (true) {
//			alpha.zero();
//			ArrayList<Integer> pList = new ArrayList<Integer>();
//			for (int i = 0; i < phiNum; i++)
//				pList.add(i);
//			int N = rand.nextInt(phiNum);
//			for (int i = 0; i < N; i++) {
//				int j = rand.nextInt(pList.size());
//				int k = pList.get(j);
//				double r = rand.nextDouble() * (R_MAX - R_MIN) + R_MIN;
//				if (r != 0) alpha.set(k, r);
//				//alpha.set(k, rand.nextInt(1) * (R_MAX - R_MIN) + R_MIN);
//				pList.remove(j);
//			}
//			boolean bExit = false;
//			double r0 = alpha.get(0);
//			for (int i = 0; i < alpha.size(); i++) {
//				if (r0 != alpha.get(i)) {
//					bExit = true;
//					break;
//				}
//			}
//			if (bExit) break;
//		}
		return alpha;
	}
			
	// calculate feature expectation mu
	public static Vector calFeatExp(FSC fsc, Matrix occ, 
			BasisFunctions phi, boolean useSparse) {
		int phiNum = phi.getNBasis();
		Vector mu = Mtrx.Vec(phiNum, useSparse);
		for (int i = 0; i < phiNum; i++) {
			double x = 0;
			for (Iterator<MatrixEntry> it = Mtrx.Iter(occ); it.hasNext(); ) {
				MatrixEntry me = it.next();
				int s = me.row();
				int n = me.column();
				int a = fsc.getAction(n);
				x += phi.get(i, s, a) * me.get();				
			}
			if (x != 0) mu.set(i, x);
		}
		return mu;
	}
	
	// solve POMDP using PBPI with given beliefs and reward
	public static FSC getNewFsc(PomdpProblem pomdp, ArrayList<Vector> pbpiBeliefs,
			Vector[] trueReward, Vector[] learnedReward, Random rand) {
		for (int a = 0; a < pomdp.nActions; a++) {
			pomdp.R[a].zero();
			pomdp.R[a].set(learnedReward[a]);			
		}
		
		PBPI pbpi = new PBPI(pomdp, pbpiBeliefs);
		pbpi.setParams(1000, 1e-6);
		FSC fsc = pbpi.run(false, rand);
		//fsc.print();

		FSC newFsc = new FSC(pomdp, fsc);
		pbpi.delete();
		pbpi = null;
		for (int a = 0; a < pomdp.nActions; a++) {
			pomdp.R[a].zero();
			pomdp.R[a].set(trueReward[a]);
		}
		return newFsc;
	}
	
	// calcalute w * (mu_E - mu)
	public static double calDiffV(Vector W, Vector muE, Vector mu) {
		double d = 0;
		for (Iterator<VectorEntry> it = Mtrx.Iter(W); it.hasNext(); ) {
			VectorEntry ve = it.next();
			int i = ve.index();
			d += ve.get() * (muE.get(i) - mu.get(i));
		}		
		return Math.abs(d);
	}
		
	// calculate V^pi(R)
	public static double calTrueV(FSC fsc, Matrix occ, Vector[] trueR) {
		double V = 0;
		Iterator<MatrixEntry> it = Mtrx.Iter(occ);
		while (it.hasNext()) {
			MatrixEntry me = it.next();
			int s = me.row();
			int n = me.column();
			int a = fsc.getAction(n);
			V += trueR[a].get(s) * me.get();
		}
		return V;
	}
		
	// calculate occupancy distribution using CPLEX
	public static double[][] calOccSA(PomdpProblem pomdp, FSC fsc) throws Exception {
		Matrix occ = fsc.calOccDist();
				
		double[][] occSA = new double[pomdp.nStates][pomdp.nActions];
		double sum = 0;
		Iterator<MatrixEntry> it = Mtrx.Iter(occ);
		while (it.hasNext()) {
			MatrixEntry me = it.next();
			int s = me.row();
			int n = me.column();
			occSA[s][fsc.getAction(n)] += me.get();
			sum += me.get();
		}
		return occSA;
	}
	
	// calculate |R-R'|_w,1
	// reward functions are normalized
//	public static double calWeightedNorm(PomdpProblem pomdp, 
//			Vector[] trueR, Vector[] learnedR, double[][] occSA) {		
//		double[][] R1 = new double[pomdp.nStates][pomdp.nActions];
//		double maxR1 = Double.NEGATIVE_INFINITY;
//		double minR1 = Double.POSITIVE_INFINITY;
//		double sum1 = 0;
//		
//		double[][] R2 = new double[pomdp.nStates][pomdp.nActions];
//		double maxR2 = Double.NEGATIVE_INFINITY;
//		double minR2 = Double.POSITIVE_INFINITY;
//		double sum2 = 0;
//		
//		for (int s = 0; s < pomdp.nStates; s++) {
//			for (int a = 0; a < pomdp.nActions; a++) {				
//				maxR1 = Math.max(maxR1, trueR[a].get(s));
//				minR1 = Math.min(minR1, trueR[a].get(s));
//				maxR2 = Math.max(maxR2, learnedR[a].get(s));
//				minR2 = Math.min(minR2, learnedR[a].get(s));
//			}
//		}
//		for (int s = 0; s < pomdp.nStates; s++) {
//			for (int a = 0; a < pomdp.nActions; a++) {
//				R1[s][a] = (trueR[a].get(s) - minR1) / (maxR1 - minR1) + 1e-6;
//				sum1 += R1[s][a];
//				R2[s][a] = (learnedR[a].get(s) - minR2) / (maxR2 - minR2) + 1e-6;
//				sum2 += R2[s][a];
//			}
//		}
//		
//		for (int s = 0; s < pomdp.nStates; s++) {
//			for (int a = 0; a < pomdp.nActions; a++) {
//				R1[s][a] /= sum1;
//				R2[s][a] /= sum2;
//			}
//		}
//
//		double norm = 0;
//		for (int s = 0; s < pomdp.nStates; s++) {
//			for (int a = 0; a < pomdp.nActions; a++) {
//				double x = R1[s][a] * Math.log(R1[s][a] / R2[s][a]);
//				double y = R2[s][a] * Math.log(R2[s][a] / R1[s][a]);
//				norm += x + y;
//			}
//		}
//		return norm / 2;
//	}
	
	// calculate |R-R'|_2
	// reward functions are normalized
	public static double calWeightedNorm(PomdpProblem pomdp, 
			Vector[] trueR, Vector[] learnedR, double[][] occSA) {		
		double[][] R1 = new double[pomdp.nStates][pomdp.nActions];
		double maxR1 = Double.NEGATIVE_INFINITY;
		double minR1 = Double.POSITIVE_INFINITY;
		
		double[][] R2 = new double[pomdp.nStates][pomdp.nActions];
		double maxR2 = Double.NEGATIVE_INFINITY;
		double minR2 = Double.POSITIVE_INFINITY;
		
		for (int s = 0; s < pomdp.nStates; s++) {
			for (int a = 0; a < pomdp.nActions; a++) {				
				maxR1 = Math.max(maxR1, trueR[a].get(s));
				minR1 = Math.min(minR1, trueR[a].get(s));
				maxR2 = Math.max(maxR2, learnedR[a].get(s));
				minR2 = Math.min(minR2, learnedR[a].get(s));
			}
		}
		for (int s = 0; s < pomdp.nStates; s++) {
			for (int a = 0; a < pomdp.nActions; a++) {
				R1[s][a] = trueR[a].get(s) - minR1;
				if (maxR1 != minR1) R1[s][a] /= (maxR1 - minR1);
				R2[s][a] = learnedR[a].get(s) - minR2;
				if (maxR2 != minR2) R2[s][a] /= (maxR2 - minR2);
			}
		}

		double norm = 0;
		for (int s = 0; s < pomdp.nStates; s++) {
			for (int a = 0; a < pomdp.nActions; a++) {
				double x = Math.abs(R1[s][a] - R2[s][a]);
				norm += x * x;
			}
		}
		return Math.sqrt(norm);
	}
	
	public static double getGoodReturn(String probName, boolean useSparse) throws Exception {
		int maxIters = 1000;
		double minConvError = 1e-5;
		
		String pomdpFileName = "./problems/" + probName + ".pomdp";
		String fscFileName = "./output/" + probName + ".pbpi";
		PomdpProblem pomdp = PomdpFile.read(pomdpFileName, useSparse);
		FSC fsc = new FSC(pomdp);
		fsc.read(fscFileName);
		fsc.evaluation(maxIters, minConvError);
		double goodReturn = fsc.getV0();
		
		fsc.delete();
		fsc = null;
		pomdp.delete();
		pomdp = null;
		return goodReturn;
	}
		
	public static int[][] combination(int n, int r) {
		int total = (int) Math.pow(n, r);
//		System.out.printf("Allocate %d^%d * %d = %d\n", n, r, r, total * r);
		int[][] result = new int[total][r];
		for (int i = 0; i < total; i++) {
			int tmp = i;
			for (int j = 0; j < r; j++) {
				int a = tmp % n;
				tmp = (int)((double)tmp / (double)n);
				result[i][j] = a;
			}
		}
		return result;
	}
	
	public static double getSum(Vector[] R) {
		double sum = 0;
		for (int a = 0; a < R.length; a++) {
			for (int s = 0; s < R[a].size(); s++) {
				sum += Math.abs(R[a].get(s));
			}
		}
		return sum;
	}
	
	public static boolean equal(Vector a, Vector b) {
		if (a.size() != b.size()) return false;
		for (int i = 0; i < a.size(); i++)
			if (a.get(i) != b.get(i)) return false;
		return true;
	}
	
	public static void print(PomdpProblem pomdp, ArrayList<FscNode> fsc) {
		Iterator<FscNode> it = fsc.iterator();
		while (it.hasNext()) {
			FscNode node = it.next();
			System.out.printf("n%d: %s\n", node.id, pomdp.actions[node.act]);
			System.out.print("    ");
			for (int obs = 0; obs < pomdp.nObservations; obs++) 
				System.out.printf("%s->n%d ", pomdp.observations[obs], node.nextNode[obs]);
			System.out.println();
		}
		System.out.println();
	}
	
	public static void print(ArrayList<Vector>[] nodeBelief) {
		int num = 0;
		for (int n = 0; n < nodeBelief.length; n++) {
			System.out.printf("=== Belief at Node %d ==========================\n", n);
			Iterator<Vector> itB = nodeBelief[n].iterator();
			for (int i = 0; itB.hasNext(); i++) {
				Vector B = itB.next();
				System.out.printf("%d : ", i);
				for (int s = 0; s < B.size(); s++)
					System.out.printf("%f ", B.get(s));
				System.out.println();
				num++;
			}
		}
		System.out.println();
		System.out.printf("# of sampled beliefs: %d\n", num);
	}
}
