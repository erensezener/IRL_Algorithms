package kr.ac.kaist.irl.fromTraj;

import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.io.BufferedWriter;
import java.io.FileWriter;
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
 * for partially observable environments using the max-margin between feature expectations method
 * when the behavior data is given by trajectories of the expert's executed actions and the corresponding observations.
 *  
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, IJCAI 2009.
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, JMLR 12, 2011.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class TrajIrlQp {
	final private double INF = 1e5;
	final private double R_MAX = 1;	
	final private double R_MIN = -1;
	
	// variables of POMDP
	private PomdpProblem pomdp;		
	private int nStates;
	private int nActions;
	private int nObservs;
	private double gamma;
	private boolean useSparse;
	private Vector[] trueR;
	private Vector[] learnedR;

	// variables of IRL
	private int nTrajs;
	private int nSteps;
	private double epsilon;
	private int maxIter;
	private BasisFunctions phi;
	private int phiNum;
	public double irlTime;
	public double trajExpV;
	public double diffV;

	private FSC optFsc;
	private double[][] optFscOccSA;
	public Vector muE;
	private Vector W;
	private ArrayList<Vector> WList;
	private ArrayList<Vector> muList;
	public ArrayList<Double> epsList;
	public ArrayList<Double> diffMuList;
	public ArrayList<Double> diffVList;
	public ArrayList<Double> trueVList;
	public ArrayList<Double> diffRList;
		
	private Random rand;
	private IloCplex cplex;

	// variables of PBPI
	private int nBeliefs;
	private ArrayList<Vector> pbpiBeliefs;
	public double totalTime;
	public double pbpiTime;	
	
	public TrajIrlQp(PomdpProblem pomdpProb) throws Exception {		
		pomdp = pomdpProb;
		nStates = pomdpProb.nStates;
		nActions = pomdpProb.nActions;
		nObservs = pomdpProb.nObservations;
		gamma = pomdpProb.gamma;
		useSparse = pomdpProb.useSparse;
		
		trueR = new Vector[nActions];
		learnedR = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			trueR[a] = Mtrx.Vec(nStates, useSparse);
			learnedR[a] = Mtrx.Vec(nStates, useSparse);
			for (int s = 0; s < nStates; s++)
				if (pomdpProb.R[a].get(s) != 0)
					trueR[a].set(s, pomdpProb.R[a].get(s));
		}
	}
	
	public void setParams(String fscFname, BasisFunctions _phi, int _nTrajs, int _nSteps,
			double _eps, int _maxIter, int _nBeliefs, double minDist, Random _rand) throws Exception {		
		phi = _phi;
		phiNum = phi.getNBasis();
		nTrajs = _nTrajs;
		nSteps = _nSteps;
		epsilon = _eps;
		maxIter = _maxIter;		
		nBeliefs = _nBeliefs;
		rand = _rand;
		muE = Mtrx.Vec(phiNum, useSparse);
		W = Mtrx.Vec(phiNum, useSparse);

		WList = new ArrayList<Vector>();
		muList = new ArrayList<Vector>();
		epsList = new ArrayList<Double>();
		diffMuList = new ArrayList<Double>();
		diffVList = new ArrayList<Double>();
		trueVList = new ArrayList<Double>();
		diffRList = new ArrayList<Double>();

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
		
		// generate the set of sampled belief points for PBPI
		int maxRestart = 10;
		pbpiBeliefs = BeliefPoints.initBeliefs(pomdp, nBeliefs, maxRestart, minDist, rand);
		System.out.printf("  # of basis functions           : %d\n", phi.getNBasis());
		System.out.printf("  # of sampled beliefs for PBPI  : %d\n", pbpiBeliefs.size());
		
		// find the optimal policy by PBPI
		pbpiTime = 0;
		optFsc = new FSC(pomdp);
		optFsc.read(fscFname);
		//optFsc = getNewFsc();
		//optFsc.print();
		//optFsc.printValue();
		optFscOccSA = IrlUtil.calOccSA(pomdp, optFsc);
				
		//mkTraj();
		mkTraj2();
	}
		
/////////////////////////////////////////////////////////////////////////////////	
	
	public void solve(boolean bPrint) throws Exception {						
		double T0 = CpuTime.getCurTime();
		
		initW();
		double T1 = CpuTime.getCurTime();
		FSC newFsc = IrlUtil.getNewFsc(pomdp, pbpiBeliefs, trueR, learnedR, rand);
		double fscTime = CpuTime.getElapsedTime(T1);
		pbpiTime += fscTime;
		double qpTime = 0;
		
		T1 = CpuTime.getCurTime();
		Matrix occ = null;
		occ = newFsc.calOccDist();
		double occTime = CpuTime.getElapsedTime(T1);
		
		Vector mu = IrlUtil.calFeatExp(newFsc, occ, phi, useSparse);
		double eps = W.norm(Vector.Norm.Two); //Double.POSITIVE_INFINITY;
		double diffMu = Mtrx.calL2Dist(muE, mu);
		double diffV = IrlUtil.calDiffV(W, muE, mu);
		double trueV = IrlUtil.calTrueV(newFsc, occ, trueR);
		double diffR = IrlUtil.calWeightedNorm(pomdp, trueR, learnedR, optFscOccSA);
		muList.add(mu);
		epsList.add(eps);
		diffMuList.add(diffMu);
		diffVList.add(diffV);
		trueVList.add(trueV);
		diffRList.add(diffR);
		occ = null;
		
		if (bPrint) {
			System.out.println("=== Start to solve ==============================================");
			System.out.println(" Iter |     Eps          diff(mu)   diff(V;R')      V^pi(R)     |R-R'| " +
					"::   Qp     Pbpi   Occ    Fsc");
			System.out.printf(" %4d | %12.6f %12.6f %12.6f %12.6f %10.4f " +
					":: %6.2f %6.2f %6.2f %5d\n", 
					0, eps, diffMu, diffV, trueV, diffR, 
					0.0, fscTime, occTime, newFsc.size());
		}
		
		for (int t = 1; t < maxIter && diffMu > epsilon; t++) {
			T1 = CpuTime.getCurTime();
			eps = calW(t);
			qpTime = CpuTime.getElapsedTime(T1);
			
			T1 = CpuTime.getCurTime();
			newFsc = IrlUtil.getNewFsc(pomdp, pbpiBeliefs, trueR, learnedR, rand);
			fscTime = CpuTime.getElapsedTime(T1);
			pbpiTime += fscTime;
			
			T1 = CpuTime.getCurTime();
			occ = newFsc.calOccDist();
			occTime = CpuTime.getElapsedTime(T1);
						
			mu = IrlUtil.calFeatExp(newFsc, occ, phi, useSparse);
			diffMu = Mtrx.calL2Dist(muE, mu);
			diffV = IrlUtil.calDiffV(W, muE, mu);
			trueV = IrlUtil.calTrueV(newFsc, occ, trueR);
			diffR = IrlUtil.calWeightedNorm(pomdp, trueR, learnedR, optFscOccSA);
								
			if (bPrint)
				System.out.printf(" %4d | %12.6f %12.6f %12.6f %12.6f %10.4f " +
						":: %6.2f %6.2f %6.2f %5d\n", 
						t, eps, diffMu, diffV, trueV, diffR,
						qpTime, fscTime, occTime, newFsc.size());
						
			//printW();
			muList.add(mu);
			epsList.add(eps);
			diffMuList.add(diffMu);
			diffVList.add(diffV);
			trueVList.add(trueV);
			diffRList.add(diffR);
			occ = null;
		}
		irlTime = CpuTime.getElapsedTime(T0);		
		System.out.printf("Elapsed time: %f sec, # of phi: %d\n", irlTime, phiNum);
		System.out.printf("   Pbpi time: %f sec (%f)\n\n", pbpiTime, pbpiTime / irlTime);
	}
	
	private double calW(int iter) throws Exception {
		Hashtable<String, IloNumVar> varMap = new Hashtable<String, IloNumVar>();
		varMap.put(strEps(), cplex.numVar(-INF, INF, strEps()));
		for (int i = 0; i < phiNum; i++) {
			String name = strW(i);
			varMap.put(name, cplex.numVar(R_MIN, R_MAX, name));
		}
		
		// ||w||_2 <= 1
		IloNumExpr expW = cplex.numExpr(); 
		for (int i = 0; i < phiNum; i++) {
			IloNumVar varW = varMap.get(strW(i));
			expW = cplex.sum(expW, cplex.prod(varW, varW));
		}
		cplex.addLe(expW, 1);
		
		// w^T mu_E >= w^T mu(j) + eps
		for (int j = 0; j < muList.size(); j++) {
			IloNumExpr expL = cplex.numExpr();
			for (Iterator<VectorEntry> it = Mtrx.Iter(muE); it.hasNext(); ) {
				VectorEntry ve = it.next();
				IloNumVar varW = varMap.get(strW(ve.index()));
				expL = cplex.sum(expL, cplex.prod(ve.get(), varW));
			}
			IloNumExpr expR = varMap.get(strEps());
			for (Iterator<VectorEntry> it = Mtrx.Iter(muList.get(j)); it.hasNext(); ) {
				VectorEntry ve = it.next();
				IloNumVar varW = varMap.get(strW(ve.index()));
				expR = cplex.sum(expR, cplex.prod(ve.get(), varW));
			}
			cplex.addGe(expL, expR);			
		}
		
		// max eps
		cplex.addMaximize(varMap.get(strEps()));
		
		// solve lp
		if (cplex.solve()) {
			/*System.out.println("-- Result of CPLEX --");
			System.out.println("Objective value = " + cplex.getObjValue());
			System.out.println("Solution status = " + cplex.getStatus());
			for (int i = 0; i < vars.length; i++) 
				System.out.println(vars[i].getName() + " = " + cplex.getValue(vars[i]));*/
		}
		else {
			System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
			System.out.println("Objective value = " + cplex.getObjValue());
			System.out.println("Solution status = " + cplex.getStatus());
			
			System.out.println("Is primal feasible? " + cplex.isPrimalFeasible());
			System.out.println("Is dual feasible? " + cplex.isDualFeasible());
			
			IloCplex.Quality inf = cplex.getQuality(IloCplex.QualityType.MaxPrimalInfeas);
			double maxinfeas = inf.getValue();
			System.out.printf("Solution quality : %.20f\n", maxinfeas);
		}
				
		double obj = cplex.getObjValue();
		W.zero();
		for (int i = 0; i < phiNum; i++) {
			double w = cplex.getValue(varMap.get(strW(i)));
			if (w != 0) W.set(i, w);
		}
		WList.add(W.copy());
		
		// save reward 
		for (int a = 0; a < nActions; a++) {
			learnedR[a].zero();
			for (int s = 0; s < nStates; s++) {
				double r = 0;
				for (Iterator<VectorEntry> it = Mtrx.Iter(W); it.hasNext(); ) {
					VectorEntry ve = it.next();
					r += ve.get() * phi.get(ve.index(), s, a);
				}
				if (r != 0) learnedR[a].set(s, r);
			}			
		}
		
		varMap.clear();
		varMap = null;
		cplex.clearModel();
		return obj;
	}
	
	private String strW(int p) { return "w[" + p + "]"; }
	private String strEps() { return "eps"; }
//	private String strR2(int s, int a) { return "R2[" + s + "][" + a + "]"; }
	
	private void initW() {
		W.set(IrlUtil.initialWeight(phiNum, R_MIN, R_MAX, useSparse, rand));
		Mtrx.scale(W, 1.0 / Math.sqrt(W.norm(Vector.Norm.One)));
		WList.add(W.copy());	
			
		// save reward 
		for (int a = 0; a < nActions; a++) {
			learnedR[a].zero();
			for (int s = 0; s < nStates; s++) {
				double r = 0;
				for (Iterator<VectorEntry> it = Mtrx.Iter(W); it.hasNext(); ) {
					VectorEntry ve = it.next();
					r += ve.get() * phi.get(ve.index(), s, a);
				}
				if (r != 0) learnedR[a].set(s, r);
			}			
		}
	}
	
/////////////////////////////////////////////////////////////////////////////////	
	
	private void mkTraj() throws Exception {
		double T0 = CpuTime.getCurTime();
		System.out.println("--- Generate trajectories and expert's feature expectation");
		trajExpV = 0;
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
					int s2 = veB.index();
					for (int i = 0; i < phiNum; i++) {
						double x = muE.get(i) + Math.pow(gamma, t) * veB.get() * phi.get(i, s2, a);
						if (x != 0) muE.set(i, x);
					}					
				}	
				
				s = pomdp.sampleNextState(s, a, rand);
				z = pomdp.sampleObserv(a, s, rand);								
				n = optFsc.getNextNode(n, z);
				B = pomdp.getNextBelief(B, a, z);
			}
		}
		trajExpV /= nTrajs;
		Mtrx.scale(muE, 1.0 / nTrajs);
		Mtrx.compact(muE);

		Matrix occ = optFsc.calOccDist();		
		Vector muOpt = IrlUtil.calFeatExp(optFsc, occ, phi, useSparse);
		double diff = Mtrx.calL2Dist(muE, muOpt);
		
		System.out.printf("  Elapsed time                   : %.4f sec\n", CpuTime.getElapsedTime(T0));
		System.out.printf("  Expected Value of Trajectories : %f (%f)\n", trajExpV, optFsc.getV0());
		System.out.printf("  Diff. of feature expectation   : %f\n", diff);
		System.out.printf("  # of nodes of optimal fsc      : %d\n", optFsc.size());
	}
	
	private void mkTraj2() throws Exception {
		double T0 = CpuTime.getCurTime();
		System.out.println("--- Generate trajectories and expert's feature expectation");
		trajExpV = 0;
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
		
		for (int i = 0; i < phiNum; i++) {
			double x = 0;
			for (int s = 0; s < nStates; s++)
				for (int a = 0; a < nActions; a++)
					x += coefs[s][a] * phi.get(i, s, a);
			if (x != 0) muE.set(i, x / nTrajs);
		}

		Matrix occ = optFsc.calOccDist();		
		Vector muOpt = IrlUtil.calFeatExp(optFsc, occ, phi, useSparse);
		diffV = Mtrx.calL2Dist(muE, muOpt);
		
		System.out.printf("  Elapsed time                   : %.4f sec\n", CpuTime.getElapsedTime(T0));
		System.out.printf("  Expected Value of Trajectories : %f (%f)\n", trajExpV, optFsc.getV0());
		System.out.printf("  Diff. of feature expectation   : %f\n", diffV);
		System.out.printf("  # of nodes of optimal fsc      : %d\n", optFsc.size());
		//diffV = Math.abs(trajExpV - optFsc.getV0());
	}
		
	// check i-th reward has same value for all elements
	public boolean checkReward(int i) {
		Vector tmpW = WList.get(i);
		double[][] tmpR = new double[nStates][nActions];
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				for (int p = 0; p < phiNum; p++) {
					tmpR[s][a] += tmpW.get(p) * phi.get(p, s, a);
				}
			}
		}
		double t = tmpR[0][0];
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				if (tmpR[s][a] != t)
					return false;
			}
		}
		return true;
	}
	
//	public double calTheta() {
//		Vector W0 = WList.get(0);
//		double a = Mtrx.dot(W0, W);
//		double b = W0.norm(Vector.Norm.Two);
//		double c = W.norm(Vector.Norm.Two);
//		return Math.acos(a / b / c) * 180.0 / Math.PI;
//	}
	
/////////////////////////////////////////////////////////////////////////////////	
	
	public void delete() throws Exception {
		if (cplex != null) {
			cplex.end();
			cplex = null;
		}
		trueR = null;
		muE = null;
		W = null;
		WList.clear(); WList = null;
		muList.clear(); muList = null;
		epsList.clear(); epsList = null;
		trueVList.clear(); trueVList = null;
		pbpiBeliefs.clear(); pbpiBeliefs = null;	
		optFsc.delete(); optFsc = null;		
	}	

	public void printW() {
		System.out.print("       W = [");
		for (int i = 0; i < phiNum; i++) 
			System.out.printf("%f ", W.get(i));
		System.out.println("]");
	}
	
	public void printResult() throws Exception {
		System.out.println("=== Reward ======================================================");
		Vector W = WList.get(WList.size() - 1);
		for (int a = 0; a < nActions; a++) {
			pomdp.R[a].zero();
			for (int s = 0; s < nStates; s++) {
				double r = 0.0;
				for (Iterator<VectorEntry> it = Mtrx.Iter(W); it.hasNext(); ) {
					VectorEntry ve = it.next();
					r += ve.get() * phi.get(ve.index(), s, a);
				}
				if (r != 0.0) pomdp.R[a].set(s, r);
				System.out.printf("   R: %s : %s : * : * %14.12f\n", 
						pomdp.actions[a], pomdp.states[s], r);
			}	
		}
		
		FSC lastFsc = IrlUtil.getNewFsc(pomdp, pbpiBeliefs, trueR, learnedR, rand);
		if (lastFsc.size() < 30) lastFsc.print();
		lastFsc.simulate(trueR, nTrajs, nSteps, false, rand);
		lastFsc.delete();
		lastFsc = null;
	}
	
	public void saveResult(String filename) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
		bw.write("result = [\n");
		for (int i = 0; i < trueVList.size(); i++) {
			String str = String.format("%d, %14.10f\n",	i, trueVList.get(i));
			bw.write(str);
		}		
		bw.write("];\n");
		bw.close();
	}
}
