package kr.ac.kaist.irl.fromTraj;

import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Random;

import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;

import kr.ac.kaist.pomdp.data.*;
import kr.ac.kaist.utils.CpuTime;
import kr.ac.kaist.utils.IrlUtil;
import kr.ac.kaist.utils.Mtrx;

/**
 * Solve inverse reinforcement learning (IRL) problems 
 * for partially observable environments using the max-margin between values method
 * when the behavior data is given by trajectories of the expert's executed actions and the corresponding observations.
 * 
 * maximize t - lambda ||R||_1
 * subject to alpha*mu_E >= alpha*mu_pi + t for all pi
 *            |alpha_i| <= 1 for all i
 * 
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, IJCAI 2009.
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, JMLR 12, 2011.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class TrajIrlLp2 {
	final private double INF = 1e5;
	final private double R_MAX = 1;	
	final private double R_MIN = -1;
	private double LAMBDA;

	private BasisFunctions phi;
	private int phiNum;
	private int nIters;
	private int nSampledBeliefs;
	private int nTrajs;
	private int nSteps;
	private double eps;

	private FSC optFsc;
	private double[][] optFscOccSA;
	private PomdpProblem pomdp;		
	private int nStates;
	private int nActions;
	private int nObservs;
	private double gamma;
	private boolean useSparse;
	
	private ArrayList<Vector> muList;
	private ArrayList<Vector> pbpiBeliefs;
	public ArrayList<Vector[]> rewardList;
	public ArrayList<Double> epsList;
	public ArrayList<Double> diffVList;
	public ArrayList<Double> trueVList;
	public ArrayList<Double> diffRList;

	private Vector alpha;
	private Vector[] learnedR;
	private Vector[] trueR;
	private Vector optFeatExp;
	private Random rand;
	
	private IloCplex cplex;
	private Hashtable<String, IloNumVar> varMap;
	private int nRows;
	private int nCols;
	
	public double totalIrlTime;
	public double totalPbpiTime;
	public double diffV;
	
	public TrajIrlLp2(PomdpProblem _pomdpProb) {		
		pomdp = _pomdpProb;		
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		nObservs = pomdp.nObservations;
		gamma = pomdp.gamma;
		useSparse = pomdp.useSparse;
	}
	
	public void setParams(String fscFname, BasisFunctions _phi, int _nIters, 
			int _nSampledBeliefs, double _eps, 
			int _nTrajs, int _nSteps, double minDist, Random _rand) throws Exception {
		phi = _phi;
		phiNum = phi.getNBasis();
		nIters = _nIters;
		nSampledBeliefs = _nSampledBeliefs;
		eps = _eps;
		nTrajs = _nTrajs;
		nSteps = _nSteps;
		rand = _rand;		

		System.out.printf("  # of basis functions           : %d\n", phi.getNBasis());

		alpha = Mtrx.Vec(phiNum, useSparse);
		learnedR = new Vector[nActions];
		trueR = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			learnedR[a] = Mtrx.Vec(nStates, useSparse);
			learnedR[a].set(pomdp.R[a]);
			trueR[a] = Mtrx.Vec(nStates, useSparse);
			trueR[a].set(pomdp.R[a]);
		}
		
		cplex = new IloCplex();	
		cplex.setOut(null);
		
		// use Dual simplex algorithm
		//cplex.setParam(IloCplex.IntParam.RootAlg, IloCplex.Algorithm.Dual);
		// set Markowitz tolerance. default: 0.01 (0.00001 ~ 0.99999)
		cplex.setParam(IloCplex.DoubleParam.EpMrk, 0.99999);
		// set optimality tolerance. default: 1e-6 (1e-9 ~ 1e-1)
		cplex.setParam(IloCplex.DoubleParam.EpOpt, 1e-9);
		// set feasibility tolerance. default: 1e-6 (1e-9 ~ 1e-1)
		cplex.setParam(IloCplex.DoubleParam.EpRHS, 1e-9);
		
		int maxRestart = 10;
		pbpiBeliefs = BeliefPoints.initBeliefs(pomdp, nSampledBeliefs, maxRestart, minDist, rand);
		System.out.printf("  # of sampled beliefs for PBPI  : %d\n", pbpiBeliefs.size());
		
		//optFsc = getNewFsc();
		optFsc = new FSC(pomdp);
		optFsc.read(fscFname);
		//optFsc.print();
		optFscOccSA = IrlUtil.calOccSA(pomdp, optFsc);
		
		optFeatExp = mkTraj2();
		rewardList = new ArrayList<Vector[]>();
		muList = new ArrayList<Vector>();
		epsList = new ArrayList<Double>();
		diffVList = new ArrayList<Double>();
		trueVList = new ArrayList<Double>();
		diffRList = new ArrayList<Double>();
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	public ArrayList<Double> solve(double lambda, boolean bPrint) throws Exception {		
		double T0 = CpuTime.getCurTime();		
		LAMBDA = lambda;
		
		// initialize
		initReward();
		double T1 = CpuTime.getCurTime();
		FSC newFsc = IrlUtil.getNewFsc(pomdp, pbpiBeliefs, trueR, learnedR, rand);
		Matrix occ = newFsc.calOccDist();
		muList.add(IrlUtil.calFeatExp(newFsc, occ, phi, useSparse));
		double fscTime = CpuTime.getElapsedTime(T1);
		totalPbpiTime += fscTime;
		double lpTime = 0;
		
		T1 = CpuTime.getCurTime();
		double diffV = calDiffExpV(newFsc, learnedR);
		newFsc.evaluation(trueR);
		double trueV = newFsc.getV0();
		double rewardNorm = IrlUtil.getSum(learnedR);
		double diffR = IrlUtil.calWeightedNorm(pomdp, trueR, learnedR, optFscOccSA);
		epsList.add(-1.0);
		diffVList.add(diffV);
		trueVList.add(trueV);
		diffRList.add(diffR);
		double evalTime = CpuTime.getElapsedTime(T1);
		
		if (bPrint) {
			System.out.println("=== Start to solve ==============================================");
			System.out.println(" Iter |    Obj Value   diff(V;R')     V^pi(R)      |R|        |R-R'| " +
					"::   Lp     Pbpi   Eval  ");
			System.out.printf(" %4d | %12.6f %12.6f %12.6f %10.4f %10.4f " +
					":: %6.2f %6.2f %6.2f (n%d, r%d, c%d)\n", 
					0, 0.0, diffV, trueV, rewardNorm, diffR,  
					lpTime, fscTime, evalTime, newFsc.size(), nRows, nCols);
		}
		
		for (int t = 1; t < nIters && diffV > eps; t++) {
			T1 = CpuTime.getCurTime();
			double obj = calNewReward(t);
			lpTime = CpuTime.getElapsedTime(T1);
			
			T1 = CpuTime.getCurTime();
			newFsc = IrlUtil.getNewFsc(pomdp, pbpiBeliefs, trueR, learnedR, rand);
			occ = newFsc.calOccDist();
			muList.add(IrlUtil.calFeatExp(newFsc, occ, phi, useSparse));
			fscTime = CpuTime.getElapsedTime(T1);
			totalPbpiTime += fscTime;
			
			T1 = CpuTime.getCurTime();
			diffV = calDiffExpV(newFsc, learnedR);
			newFsc.evaluation(trueR);
			trueV = newFsc.getV0();
			rewardNorm = IrlUtil.getSum(learnedR);
			diffR = IrlUtil.calWeightedNorm(pomdp, trueR, learnedR, optFscOccSA);
			epsList.add(obj);
			diffVList.add(diffV);
			trueVList.add(trueV);
			diffRList.add(diffR);
			evalTime = CpuTime.getElapsedTime(T1);

			if (bPrint) {
				System.out.printf(" %4d | %12.6f %12.6f %12.6f %10.4f %10.4f " +
						":: %6.2f %6.2f %6.2f (n%d, r%d, c%d)\n", 
						t, obj, diffV, trueV, rewardNorm, diffR,  
						lpTime, fscTime, evalTime, newFsc.size(), nRows, nCols);
			}
			
			//printReward();
			//newFsc.print();
		}
		totalIrlTime = CpuTime.getElapsedTime(T0);
		System.out.printf("Elapsed Time  : %f sec\n", totalIrlTime);
		System.out.printf("   PBPI Time  : %f sec (%.2f)\n\n", 
				totalPbpiTime, totalPbpiTime / totalIrlTime * 100);
		return trueVList;
	}
	
	private double calNewReward(int t) throws Exception {
		varMap = new Hashtable<String, IloNumVar>();
		varMap.put("eps", cplex.numVar(-INF, INF, "eps"));
//		for (int pi = 0; pi < muList.size(); pi++) {
//			String name = strEps(pi);
//			varMap.put(name, cplex.numVar(-INF, INF, name));
//		}
		for (int p = 0; p < phiNum; p++) {
			String name = strAlpha(p);
			varMap.put(name, cplex.numVar(R_MIN, R_MAX, name));
		}
		
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				String name = strR2(s, a);
				varMap.put(name, cplex.numVar(0, INF, name));
			}
		}
		
		// -R2(s,a) <= sum_p alpha(p) * phi(p,s,a) <= R2(s,a)
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				IloNumExpr exp1L = cplex.prod(-1, varMap.get(strR2(s, a)));
				IloNumExpr exp1R = cplex.numExpr();
				for (int p = 0; p < phiNum; p++) {
					IloNumVar varA = varMap.get(strAlpha(p));
					exp1R = cplex.sum(exp1R, cplex.prod(phi.get(p, s, a), varA));
				}
				cplex.addLe(exp1L, exp1R);

				IloNumExpr exp2L = cplex.numExpr();
				for (int p = 0; p < phiNum; p++) {
					IloNumVar varA = varMap.get(strAlpha(p));
					exp2L = cplex.sum(exp2L, cplex.prod(phi.get(p, s, a), varA));
				}
				cplex.addLe(exp2L, varMap.get(strR2(s, a)));
			}
		}
		
		// V^*(b_0) - V^pi(b_0) >= eps
		for (int pi = 0; pi < muList.size(); pi++) {
			IloNumExpr exp = cplex.numExpr();
			Iterator<VectorEntry> it = Mtrx.Iter(optFeatExp);
			while (it.hasNext()) {
				VectorEntry ve = it.next();
				IloNumVar varA = varMap.get(strAlpha(ve.index()));
				exp = cplex.sum(exp, cplex.prod(ve.get(), varA));
			}
			Iterator<VectorEntry> it2 = Mtrx.Iter(muList.get(pi));
			while (it2.hasNext()) {
				VectorEntry ve2 = it2.next();
				IloNumVar varA = varMap.get(strAlpha(ve2.index()));
				exp = cplex.sum(exp, cplex.prod(-ve2.get(), varA));
			}
//			cplex.addGe(exp, varMap.get(strEps(pi)));
			cplex.addGe(exp, varMap.get("eps"));
		}
		
		IloNumExpr exObj = varMap.get("eps");
//		IloNumExpr exObj = cplex.numExpr();
//		for (int pi = 0; pi < muList.size(); pi++)
//			exObj = cplex.sum(exObj, varMap.get(strEps(pi)));
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				IloNumVar varR2 = varMap.get(strR2(s, a));
				exObj = cplex.sum(exObj, cplex.prod(-LAMBDA, varR2));
			}
		}
		cplex.addMaximize(exObj);
		
		// solve lp
		if (!cplex.solve())
			System.out.println("Solution status = " + cplex.getStatus());
		
		double obj = cplex.getObjValue();
		// save alpha and reward
		alpha.zero();
		for (int p = 0; p < phiNum; p++) { 
			double x = cplex.getValue(varMap.get(strAlpha(p)));
			if (x != 0) alpha.set(p, x);
		}
		Vector[] R = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			R[a] = Mtrx.Vec(nStates, useSparse);
			learnedR[a].zero();
			for (int s = 0; s < nStates; s++) {
				double x = 0.0;
				Iterator<VectorEntry> it = Mtrx.Iter(alpha);
				while (it.hasNext()) {
					VectorEntry ve = it.next();
					int p = ve.index();
					x += ve.get() * phi.get(p, s, a);
				}
				if (x != 0.0) {
					learnedR[a].set(s, x);
					R[a].set(s, x);
				}
			}
		}
		rewardList.add(R);	
		nRows = cplex.getNrows();
		nCols = cplex.getNcols();
		varMap.clear();
		varMap = null;
		cplex.clearModel();
		return obj;
	}

	private String strR2(int s, int a) { return "R2[" + s + "][" + a + "]"; };
	private String strAlpha(int p) { return "alpha[" + p + "]"; }
	private String strEps(int pi) { return "eps[" + pi + "]"; }
	
	private void initReward() {
		alpha.zero();
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
		
		Vector[] R = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			R[a] = Mtrx.Vec(nStates, useSparse);
			learnedR[a].zero();
			for (int s = 0; s < nStates; s++) {
				double x = 0;
				Iterator<VectorEntry> it = Mtrx.Iter(alpha);
				while (it.hasNext()) {
					VectorEntry ve = it.next();
					int p = ve.index();
					x += ve.get() * phi.get(p, s, a);
				}
				if (x != 0) {
					learnedR[a].set(s, x);
					R[a].set(s, x);
				}
			}
		}
		rewardList.add(R);
	}
	
/////////////////////////////////////////////////////////////////////////////////	

	private Vector mkTraj2() throws Exception {		
		double T0 = CpuTime.getCurTime();
		System.out.println("--- Generate trajectories and expert's feature expectation");
		double trajExpV = 0;
		double[][] coefs = new double[nStates][nActions];
		for (int m = 0; m < nTrajs; m++) {			
			Vector B = pomdp.start.copy();
			int s = pomdp.sampleState(B, rand);
			int n = optFsc.getStartNode();
			int a = -1;
			int z = -1;
			//System.out.printf("%d ", m);
			//if (m % 100 == 0) System.out.println();
			for (int t = 0; t < nSteps; t++) {
				a = optFsc.getAction(n);
				trajExpV += Math.pow(gamma, t) * pomdp.R[a].get(s);
				
				Iterator<VectorEntry> itB = B.iterator();
				while (itB.hasNext()) {
					VectorEntry veB = itB.next();
					coefs[veB.index()][a] += Math.pow(gamma, t) * veB.get();	
				}	
				
				s = pomdp.sampleNextState(s, a, rand);
				z = pomdp.sampleObserv(a, s, rand);								
				n = optFsc.getNextNode(n, z);
				B = pomdp.getNextBelief(B, a, z);
			}
		}
		trajExpV /= nTrajs;
//		diffV = Math.abs(trajExpV - optFsc.getV0());
		
		Vector muE = Mtrx.Vec(phiNum, useSparse);
		for (int i = 0; i < phiNum; i++) {
			double x = 0;
			for (int s = 0; s < nStates; s++)
				for (int a = 0; a < nActions; a++)
					x += coefs[s][a] * phi.get(i, s, a);
			if (x != 0) muE.set(i, x / nTrajs);
		}

		System.out.printf("  # of nodes of optimal fsc      : %d\n", optFsc.size());
		System.out.printf("  Expected Value of Trajectories : %f (%f)\n", trajExpV, optFsc.getV0());
		try {
			Matrix occ = optFsc.calOccDist();		
			Vector muOpt = IrlUtil.calFeatExp(optFsc, occ, phi, useSparse);
			diffV = Mtrx.calL2Dist(muE, muOpt);
			System.out.printf("  Diff. of feature expectation   : %f\n", diffV);
		} catch (Exception e) {
			System.err.println(e);
		}
		System.out.printf("  Elapsed time                   : %.4f sec\n", CpuTime.getElapsedTime(T0));
		
		return muE;
	}
	
	private double calDiffExpV(FSC newFsc, Vector[] reward) {
		newFsc.evaluation(reward);
		double expV1 = newFsc.calMaxV(pomdp.start.copy());
		double expV2 = 0;
		Iterator<VectorEntry> it = Mtrx.Iter(alpha);
		while (it.hasNext()) {
			VectorEntry ve = it.next();
			int p = ve.index();
			expV2 += optFeatExp.get(p) * ve.get();
		}
		return Math.abs(expV1 - expV2);
	}
	
	// check i-th reward has same value for all elements
	public boolean checkReward(int i) {
		double t = rewardList.get(i)[0].get(0);
		for (int a = 0; a < nActions; a++) {
			for (int s = 0; s < nStates; s++) {
				if (rewardList.get(i)[a].get(s) != t)
					return false;
			}
		}
		return true;
	}
	
/////////////////////////////////////////////////////////////////////////////////	
	
	public void delete() {
		if (cplex != null) {
			cplex.end();	
			cplex = null;
		}
		
		alpha = null;
		learnedR = null;
		trueR = null;
		optFeatExp = null;	
				
		muList.clear();
		muList = null;
		
		pbpiBeliefs.clear();
		pbpiBeliefs = null;	
		
		rewardList.clear();
		rewardList =  null;
		
		optFsc.delete();
		optFsc = null;
	}

	public void printReward() {
		System.out.println("=== Reward ======================================");
		for (int a = 0; a < nActions; a++) {
			for (int s = 0; s < nStates; s++) {
				System.out.printf("   R: %s : s%d : * : * %14.12f\n", 
						pomdp.actions[a], s, learnedR[a].get(s));
//				System.out.printf("   R: %s : %s : * : * %14.12f\n",
//						pomdp.actions[a], pomdp.states[s], curR[a].get(s));
			}
		}
	}
}
