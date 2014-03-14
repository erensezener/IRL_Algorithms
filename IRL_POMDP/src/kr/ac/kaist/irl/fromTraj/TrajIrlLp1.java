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
 * Select the beliefs reachable by intermediate policies in the expert's trajectories.
 * 
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, IJCAI 2009.
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, JMLR 12, 2011.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class TrajIrlLp1 {
	final private double PENALTY_WEIGHT = 1;
	final private double INF = 1e5;
	final private double R_MAX = 1;	
	final private double R_MIN = -1;
	private double V_MAX;	
	private double V_MIN;
	private double LAMBDA;

	private BasisFunctions phi;
	private int phiNum;
	private int nIters;
	private int nRepresentatives;
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
	
	private ArrayList<FSC> PI;
	private ArrayList<int[]> belief2node;
	private ArrayList<int[]>[] trajs;
	private ArrayList<Vector> representativeBeliefs;
	private ArrayList<Vector> pbpiBeliefs;
	public ArrayList<Vector[]> rewardList;
	public ArrayList<Double> epsList;
	public ArrayList<Double> diffVList;
	public ArrayList<Double> trueVList;
	public ArrayList<Double> diffRList;

	private Vector alpha;
	private Vector[] learnedR;
	private Vector[] trueR;
	private Vector[] optFeatExp;
	private Random rand;
	
	private IloCplex cplex;
	private Hashtable<String, IloNumVar> varMap;
	private int nRows;
	private int nCols;
	
	public double totalIrlTime;
	public double totalPbpiTime;
	public double diffV;
	
	public TrajIrlLp1(PomdpProblem _pomdpProb) {		
		pomdp = _pomdpProb;		
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		nObservs = pomdp.nObservations;
		gamma = pomdp.gamma;
		useSparse = pomdp.useSparse;
		
		V_MAX = R_MAX / (1 - gamma);
		V_MIN = R_MIN / (1 - gamma);
	}
	
	public void setParams(String fscFname, BasisFunctions _phi, int _nIters, 
			int _nRepresentatives, int _nSampledBeliefs, double _eps, 
			int _nTrajs, int _nSteps, double minDist, Random _rand) throws Exception {
		phi = _phi;
		phiNum = phi.getNBasis();
		nIters = _nIters;
		nRepresentatives = _nRepresentatives;
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
				
		//optFsc = getNewFsc();
		optFsc = new FSC(pomdp);
		optFsc.read(fscFname);
		//optFsc.print();
		optFscOccSA = IrlUtil.calOccSA(pomdp, optFsc);
				
		mkTrajs();		
		optFeatExp = calFeatExp();
		rewardList = new ArrayList<Vector[]>();
		PI = new ArrayList<FSC>();
		belief2node = new ArrayList<int[]>();
		epsList = new ArrayList<Double>();
		diffVList = new ArrayList<Double>();
		trueVList = new ArrayList<Double>();
		diffRList = new ArrayList<Double>();

		int maxRestart = 10;
		pbpiBeliefs = new ArrayList<Vector>();
		for (Vector b : representativeBeliefs) pbpiBeliefs.add(b.copy());
		pbpiBeliefs = BeliefPoints.initBeliefs(pomdp, representativeBeliefs, 
				nSampledBeliefs, maxRestart, minDist, rand);
		System.out.printf("  # of sampled beliefs for PBPI  : %d\n", pbpiBeliefs.size());
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	public ArrayList<Double> solve(double lambda, boolean bPrint) throws Exception {		
		double T0 = CpuTime.getCurTime();		
		LAMBDA = lambda;
		
		// initialize
		initReward();
		double T1 = CpuTime.getCurTime();
		FSC newFsc = IrlUtil.getNewFsc(pomdp, pbpiBeliefs, trueR, learnedR, rand);
		PI.add(newFsc);
		belief2node.add(findReachableBeliefs(newFsc));
		double fscTime = CpuTime.getElapsedTime(T1);
		totalPbpiTime += fscTime;
		double lpTime = 0;
		
		T1 = CpuTime.getCurTime();
		double diffV = calDiffExpV(0, learnedR);
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
			PI.add(newFsc);	
			belief2node.add(findReachableBeliefs(newFsc));
			fscTime = CpuTime.getElapsedTime(T1);
			totalPbpiTime += fscTime;
			
			T1 = CpuTime.getCurTime();
			diffV = calDiffExpV(t, learnedR);
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
							
		setVariables();
		for (int pi = 0; pi < PI.size(); pi++) {
			setConstV(pi);
			setConstVb(pi);
		}
		setConstR2();
		setConstEps();
		setObjFunc();
		
		// solve lp
		if (!cplex.solve())
			System.out.println("Solution status = " + cplex.getStatus());
		
		double obj = cplex.getObjValue();
		// save alpha and reward
		alpha.zero();
		for (int p = 0; p < phiNum; p++) { 
			double x = cplex.getValue(varMap.get(strAlpha(p)));
			if (x != 0.0) alpha.set(p, x);
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
	
	private void setVariables() throws Exception {
		// Alpha
		for (int p = 0; p < phiNum; p++) {
			String name = strAlpha(p);
			varMap.put(name, cplex.numVar(R_MIN, R_MAX, name));
		}
		
		// V^pi(n,s)
		for (int pi = 0; pi < PI.size(); pi++) {
			for (int n = 0; n < PI.get(pi).size(); n++) {
				for (int s = 0; s < nStates; s++) {
					String name = strV(pi, n, s);
					varMap.put(name, cplex.numVar(V_MIN, V_MAX, name));
				}
			}
		}
		
		// V^pi(b)
		for (int pi = 0; pi < PI.size(); pi++) {
			for (int b = 0; b < representativeBeliefs.size(); b++){
				int n = belief2node.get(pi)[b];
				if (n != -1) {
					String name = strVb(pi, b);
					varMap.put(name, cplex.numVar(V_MIN, V_MAX, name));
				}
			}
		}

		// eps(pi, b)
		for (int pi = 0; pi < PI.size(); pi++) {
			for (int b = 0; b < representativeBeliefs.size(); b++) {
				int n = belief2node.get(pi)[b];
				if (n != -1) {
					String name = strEps(pi, b);
					varMap.put(name, cplex.numVar(-INF, INF, name));
				}
			}
		}

		// R_2(s,a)
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				String name = strR2(s, a);
				varMap.put(name, cplex.numVar(0, R_MAX, name));
			}
		}
	}	
	
	private void setConstEps() throws Exception {	
		for (int pi = 0; pi < PI.size(); pi++) {
			for (int b = 0; b < representativeBeliefs.size(); b++) {
				int n = belief2node.get(pi)[b];
				if (n != -1) {
					// v^*(b) - v^pi(b) >= eps(pi, b)
					IloNumExpr exp1 = cplex.numExpr();
					Iterator<VectorEntry> it = Mtrx.Iter(optFeatExp[b]);
					while (it.hasNext()) {
						VectorEntry ve = it.next();
						IloNumVar varA = varMap.get(strAlpha(ve.index()));
						exp1 = cplex.sum(exp1, cplex.prod(ve.get(), varA));
					}
					exp1 = cplex.sum(exp1, cplex.prod(-1, varMap.get(strVb(pi, b))));
					cplex.addGe(exp1, varMap.get(strEps(pi, b)));
			
					// PENALTY_WEIGHT * (v^*(b) - v^pi(b)) >= eps(pi, b)
					IloNumExpr exp2 = cplex.numExpr();
					Iterator<VectorEntry> it2 = Mtrx.Iter(optFeatExp[b]);
					while (it2.hasNext()) {
						VectorEntry ve = it2.next();
						IloNumVar varA = varMap.get(strAlpha(ve.index()));
						exp2 = cplex.sum(exp2, cplex.prod(ve.get(), varA));
					}
					exp2 = cplex.sum(exp2, cplex.prod(-1, varMap.get(strVb(pi, b))));
					double C = 1.0 / PENALTY_WEIGHT;
					cplex.addGe(exp2, cplex.prod(C, varMap.get(strEps(pi, b))));
				}
			}
		}
	}
	
	private void setObjFunc() throws Exception {
		IloNumExpr exObj = cplex.numExpr();
		// sum_pi,b eps(pi,b) 
		for (int pi = 0; pi < PI.size(); pi++) {			
			int cnt = 0;
			IloNumExpr exp = cplex.numExpr();
			for (int b = 0; b < representativeBeliefs.size(); b++) {
				int n = belief2node.get(pi)[b];
				if (n != -1) {
					exp = cplex.sum(exp, varMap.get(strEps(pi, b)));
					cnt++;
				}
			}
			if (cnt > 0) exObj = cplex.sum(exObj, cplex.prod(1.0 / cnt, exp));
//			exObj = cplex.sum(exObj, exp);
		}
		// -lambda*||R||_1
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				IloNumVar eps = varMap.get(strR2(s, a));
				exObj = cplex.sum(exObj, cplex.prod(-LAMBDA, eps));
			}
		}
		cplex.addMaximize(exObj);
	}
	
	private void setConstR2() throws Exception {
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
	}
			
	// V^pi(n,s) = sum_p alpha(p) phi(p,s,psi(n))  
	// + gamma sum_s' T(s,psi(n),s') sum_z O(s',psi(n),z) V^pi(eta(n,z),s')
	private void setConstV(int pi) throws Exception {
		int nNodes = PI.get(pi).size();
		for (int n = 0; n < nNodes; n++) {
			int a = PI.get(pi).getAction(n);
			for (int s = 0; s < nStates; s++) {	
				IloNumExpr exL = varMap.get(strV(pi, n, s));
				IloNumExpr exR = cplex.numExpr();
				for (int p = 0; p < phiNum; p++) {
					IloNumVar varA = varMap.get(strAlpha(p));
					exR = cplex.sum(exR, cplex.prod(phi.get(p, s, a), varA));
				}
				Iterator<VectorEntry> itT = Mtrx.Iter(pomdp.T[a][s]);
				while (itT.hasNext()) {
					VectorEntry veT = itT.next();
					int s2 = veT.index();
					double T = veT.get();
					Iterator<VectorEntry> itO = Mtrx.Iter(pomdp.O[a][s2]);
					while (itO.hasNext()) {
						VectorEntry veO = itO.next();
						int z = veO.index();
						double O = veO.get();
						int n2 = PI.get(pi).getNextNode(n, z);
						if (n2 != FscNode.NO_INFO) {
							double coef = gamma * T * O;
							IloNumVar varV2 = varMap.get(strV(pi, n2, s2));
							exR = cplex.sum(exR, cplex.prod(coef, varV2));
						}
					}
				}
				cplex.addEq(exL, exR);
				exL = null;
				exR = null;
			}
		}
	}
	
	private void setConstVb(int pi) throws Exception {
		// V^pi(b) = \sum_s b(s) V^pi(n,s)
		for (int b = 0; b < representativeBeliefs.size(); b++) {
			Vector B = representativeBeliefs.get(b);
			int n = belief2node.get(pi)[b];
			if (n != -1) {
				IloNumExpr exR = cplex.numExpr();
				Iterator<VectorEntry> itB = Mtrx.Iter(B);
				while (itB.hasNext()) {
					VectorEntry veB = itB.next();
					IloNumVar varV = varMap.get(strV(pi, n, veB.index()));
					exR = cplex.sum(exR, cplex.prod(veB.get(), varV));
				}
				cplex.addEq(varMap.get(strVb(pi, b)), exR);
			}
		}
		// V^*(b) >= V^pi(b) 
		for (int b = 0; b < representativeBeliefs.size(); b++) {
			int n = belief2node.get(pi)[b];
			if (n != -1) {
				IloNumExpr exp = cplex.numExpr();
				Iterator<VectorEntry> it = Mtrx.Iter(optFeatExp[b]);
				while (it.hasNext()) {
					VectorEntry ve = it.next();
					IloNumVar varA = varMap.get(strAlpha(ve.index()));
					exp = cplex.sum(exp, cplex.prod(ve.get(), varA));
				}
				cplex.addGe(exp, varMap.get(strVb(pi, b)));
			}
		}
	}	
	
	private String strEps(int pi, int b) { return "eps[" + pi + "][" + b + "]"; };
	private String strR2(int s, int a) { return "R2[" + s + "][" + a + "]"; };
	private String strAlpha(int p) { return "alpha[" + p + "]"; }
	private String strV(int pi, int n, int s) { return String.format("V%d[%d][%d]", pi, n, s); }
	private String strVb(int pi, int b) { return String.format("V%d[%d]", pi, b); }
	
	private void initReward() {
		alpha.set(IrlUtil.initialWeight(phiNum, R_MIN, R_MAX, useSparse, rand));
		
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
	
	private void mkTrajs() {
		double T0 = CpuTime.getCurTime();
		System.out.println("--- Generate trajectories and expert's feature expectation");
		
		trajs = new ArrayList[nTrajs];
		ArrayList<Vector> tmpList = new ArrayList<Vector>();
		ArrayList<Integer> cntList = new ArrayList<Integer>();		
		double expValue = 0;
		for (int m = 0; m < nTrajs; m++) {
			trajs[m] = new ArrayList<int[]>();
			Vector B = pomdp.start.copy();
			int s = pomdp.sampleState(B, rand);
			int n = optFsc.getStartNode();
			int a = -1;
			int z = -1;
			double value = 0;
			for (int t = 0; t < nSteps; t++) {
				int bId = Mtrx.find(B, tmpList);
				if (bId == -1) {
					tmpList.add(B.copy());
					cntList.add(1);
				}
				else cntList.set(bId, cntList.get(bId) + 1);
					
				a = optFsc.getAction(n);
				value += Math.pow(pomdp.gamma, t) * pomdp.R[a].get(s);
				s = pomdp.sampleNextState(s, a, rand);
				z = pomdp.sampleObserv(a, s, rand);
				trajs[m].add(new int[] {a, z});
				n = optFsc.getNextNode(n, z);
				B = pomdp.getNextBelief(B, a, z);
			}		
			expValue += value;
		}
		expValue /= nTrajs;
			
		representativeBeliefs = new ArrayList<Vector>();
		representativeBeliefs.add(pomdp.start.copy());
		while (representativeBeliefs.size() < nRepresentatives && cntList.size() > 0) {
			int maxId = -1;
			int maxCnt = Integer.MIN_VALUE;
			for (int i = 0; i < cntList.size(); i++) {
				int cnt = cntList.get(i);
				if (cnt > maxCnt) {
					maxId = i;
					maxCnt = cnt;
				}
			}
			Vector b = tmpList.get(maxId).copy();
			if (Mtrx.find(b, representativeBeliefs) == -1)
				representativeBeliefs.add(b);
			tmpList.remove(maxId);
			cntList.remove(maxId);			
		}
		tmpList.clear();
		cntList.clear();
		tmpList = null;
		cntList = null;
		System.out.printf("  Elapsed time                   : %.4f sec\n", CpuTime.getElapsedTime(T0));
		System.out.printf("  Expected Value of Trajectories : %f (%f)\n", expValue, optFsc.getV0());
		System.out.printf("  # of nodes of optimal fsc      : %d\n", optFsc.size());
		System.out.printf("  # of representative beliefs    : %d\n", representativeBeliefs.size());
//		diffV = Math.abs(expValue - optFsc.getV0());
	}
	
	// muE_i(b_j) = 1 / M_j sum_m sum_t^(T_j^m) gamma^(t-T_j^m) phi_i(b_t^m, a_t^m)
	private Vector[] calFeatExp() {
		System.out.println("--- Generate feature expectation");
		double T0 = CpuTime.getCurTime();
		int numOfBelief = representativeBeliefs.size();
		
		int[] cnt = new int[numOfBelief];		
		double[][][] coefs = new double[numOfBelief][nStates][nActions];
		
		for (int m = 0; m < nTrajs; m++) {
			int[] t0 = new int[numOfBelief];
			for (int i = 0; i < numOfBelief; i++) t0[i] = -1;
			
			Vector b = pomdp.start.copy();
			for (int t = 0; t < nSteps; t++) {
				int a = trajs[m].get(t)[0];
				int z = trajs[m].get(t)[1];
				
				for (int i = 0; i < numOfBelief; i++) {
					if (t0[i] == -1 && Mtrx.equal(b, representativeBeliefs.get(i))) {
						t0[i] = t;
						cnt[i]++;
					}
					if (t0[i] > -1) {
						for (Iterator<VectorEntry> itB = Mtrx.Iter(b); itB.hasNext(); ) {
							VectorEntry veB = itB.next();
							coefs[i][veB.index()][a] += 
								Math.pow(gamma, t - t0[i]) * veB.get();
						}
					}
				}				
				b = pomdp.getNextBelief(b, a, z);
			}
		}
		Vector[] result = new Vector[numOfBelief];
		for (int b = 0; b < numOfBelief; b++) {
			result[b] = Mtrx.Vec(phiNum, useSparse);
			for (int p = 0; p < phiNum; p++) {
				double x = 0;
				for (int s = 0; s < nStates; s++)
					for (int a = 0; a < nActions; a++)
						x += coefs[b][s][a] * phi.get(p, s, a);
				if (x != 0 && cnt[b] > 0)
					result[b].set(p, x / cnt[b]);
			}
		}
		
		try {
			Matrix occ = optFsc.calOccDist();		
			Vector muOpt = IrlUtil.calFeatExp(optFsc, occ, phi, useSparse);
			diffV = Mtrx.calL2Dist(result[0], muOpt);
			System.out.printf("  Diff. of feature expectation   : %f\n", diffV);
		} catch (Exception e) {
			System.err.println(e);
		}
		
		System.out.printf("  Elapsed time                   : %.4f sec\n", 
				CpuTime.getElapsedTime(T0));
		return result;
	}	
	
	private double calDiffExpV(int pi, Vector[] reward) {
//		double diff = Double.POSITIVE_INFINITY;
//		try {
//			Matrix occ = PI.get(pi).calOccDist();		
//			Vector mu = IrlUtil.calFeatExp(PI.get(pi), occ, phi, useSparse);
//			diff = Mtrx.calL2Dist(mu, optFeatExp[0]);
//		} catch (Exception e) {
//			System.err.println(e);
//		}
//		return diff;
		
//		FSC fsc = PI.get(pi);
//		fsc.evaluation(reward);
//		double v1 = 0;
//		Iterator<VectorEntry> it = Mtrx.Iter(alpha);
//		while (it.hasNext()) {
//			VectorEntry ve = it.next();
//			v1 += optFeatExp[0].get(ve.index()) * ve.get();
//		}
//		return Math.abs(v1 - fsc.calMaxV(representativeBeliefs.get(0)));
		
		FSC fsc = PI.get(pi);
		fsc.evaluation(reward);
		int cnt = 0;
		double sum = 0;
		double maxDiff = Double.NEGATIVE_INFINITY;
		for (int b = 0; b < representativeBeliefs.size(); b++) {
			int n = belief2node.get(pi)[b];
			if (n != -1) {
				double V_E = Mtrx.dot(alpha, optFeatExp[b]);
				double V_pi = fsc.calV(n, representativeBeliefs.get(b));
//				double V_pi = fsc.calMaxV(representativeBeliefs.get(b));
				double d = Math.abs(V_E - V_pi);
				maxDiff = Math.max(maxDiff, d);
				sum += d;
				cnt++;
			}
		}
		return sum / cnt;
//		return maxDiff;
	}
	
	private int[] findReachableBeliefs(FSC fsc) {
		int[] b2n = new int[representativeBeliefs.size()];
		for (int i = 0; i < b2n.length; i++) b2n[i] = -1;
		ArrayList<Vector>[] nodeBeliefs = fsc.findReachableBeliefs();
		for (int n = 0; n < fsc.size(); n++) {
			for (Vector b : nodeBeliefs[n]) {
				int k = Mtrx.find(b, representativeBeliefs);
				if (k != -1) b2n[k] = n;
			}
		}
//		for (int i = 0; i < b2n.length; i++)
//			System.out.printf("b%d : n%d, ", i, b2n[i]);
//		System.out.println();
		return b2n;
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
		
		for (int i = 0; i < trajs.length; i++) {
			trajs[i].clear();
			trajs[i] = null;
		}
		trajs = null;
		
		for (int i = 0; i < PI.size(); i++)       
			PI.get(i).delete();
		PI.clear();
		PI = null;
		
		representativeBeliefs.clear();
		representativeBeliefs = null;	
		
		pbpiBeliefs.clear();
		pbpiBeliefs = null;	
		
		rewardList.clear();
		rewardList =  null;
		
		optFsc.delete();
		optFsc = null;
	}
	
	public void printInfo() {
		for (int iFsc = 0; iFsc < PI.size(); iFsc++) {		
			System.out.printf("--- Reward (%d) ---------------------------------\n", iFsc);
			Vector[] R = rewardList.get(iFsc);
			for (int a = 0; a < nActions; a++) {
				for (int s = 0; s < nStates; s++) {
					System.out.printf("  R: %s : %s : * : * %14.12f\n", 
							pomdp.actions[a], pomdp.states[s], R[a].get(s));
				}
			}
			System.out.printf("--- FSC (%d) ------------------------------------\n", iFsc);
			System.out.printf("  StartNode: %d\n", PI.get(iFsc).getStartNode());
			for (int n = 0; n < PI.get(iFsc).size(); n++) {
				int a = PI.get(iFsc).getAction(n);
				System.out.printf("  n%d: %s\n", n, pomdp.actions[a]);
				System.out.print("      ");
				for (int obs = 0; obs < nObservs; obs++) {
					int n2 = PI.get(iFsc).getNextNode(n, obs);
					System.out.printf("%s -> n%d, ", pomdp.observations[obs], n2);
				}
				System.out.println();				
			}
		}
		System.out.println("------------------------------------------------");
		System.out.printf("True Value of Optimal FSC: %f\n", optFsc.getV0());
		System.out.printf("Total Elapsed Time: %f\n", totalIrlTime);
	}	
		
	public void printBeliefList() {
		System.out.println("--- Representative Beliefs --------------------------");
		for (int i = 0; i < representativeBeliefs.size(); i++) {
			Vector b = representativeBeliefs.get(i);
			System.out.printf("  b%d ", i);
			for (int j = 0; j < b.size(); j++)
				System.out.printf("%f ", b.get(j));
			System.out.println();
		}
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
	
	public void printBeliefCoefs() {
		for (int i = 0; i < representativeBeliefs.size(); i++) {
			System.out.printf("%d :: ", i);
			for (int j = 0; j < phiNum; j++)  
				System.out.printf("%f ", optFeatExp[i].get(j));
			System.out.println();
		}
	}
}
