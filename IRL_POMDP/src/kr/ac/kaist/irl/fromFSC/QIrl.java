package kr.ac.kaist.irl.fromFSC;

import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Random;

import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;

import kr.ac.kaist.pomdp.data.BeliefPoints;
import kr.ac.kaist.pomdp.data.FSC;
import kr.ac.kaist.pomdp.data.FscNode;
import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.pomdp.pbpi.PBPI;
import kr.ac.kaist.utils.CpuTime;
import kr.ac.kaist.utils.FindShapingFunction;
import kr.ac.kaist.utils.IrlUtil;
import kr.ac.kaist.utils.Mtrx;

/**
 * Solve inverse reinforcement learning (IRL) problems 
 * for partially observable environments using Q-function based optimality constraint
 * when the expert policy is explicitly given in the form of a finite state controller (FSC).
 *
 * (N', pi') = Q-backup(N, pi)
 * sum_s b(n,s) V^pi(n,s) >= sum_s b(n,s) V^pi'(n',s)
 * sum_s b(n,s) V^pi(n,s) - sum_s b(n,s) V^pi'(n',s) = eps(n)
 * for all n in N, all n' in N', all b in B(n)
 * obj: sum eps(n) + lambda*|R|
 *
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, IJCAI 2009.
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, JMLR 12, 2011.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class QIrl {
	public final double INF = 1e5;//Double.POSITIVE_INFINITY;
	public final double R_MAX = 1;	
	public final double R_MIN = -1;	
	private double LAMBDA;
	private double V_MAX;
	private double V_MIN;

	private PomdpProblem pomdp;		
	private int nStates;
	private int nActions;
	private int nObservs;
	private double gamma;
	private int[][] etaList;
	
	private int nNodes;
	private FSC optFsc;
	private ArrayList<Vector>[] nodeBelief;
	private double[][] optFscOccSA;
		
	private IloCplex cplex;
	private int nVars;
	private IloNumVar[] vars;
	private Hashtable<String, IloNumVar> varMap;
	public int nCols;
	public int nRows;
	
	private Vector[] trueReward;
	private double[][] learnedReward;
	private double[][] transformedReward;
	private boolean bPrint;

	// variables of PBPI
	private ArrayList<Vector> pbpiBeliefSet;
	public double trueV;
	public int fscSize;
		
	public QIrl(PomdpProblem _pomdp, FSC _fsc,
			int nBeliefs, double minDist, boolean _bPrint, Random rand) {
		pomdp = _pomdp;
		optFsc = _fsc;
		nodeBelief = _fsc.getNodeBelief();
		bPrint = _bPrint;
		
		nNodes = optFsc.size();
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		nObservs = pomdp.nObservations;
		gamma = pomdp.gamma;
		V_MAX = R_MAX / (1 - gamma);
		V_MIN = R_MIN / (1 - gamma);
		learnedReward = new double[nStates][nActions];
		trueReward = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			trueReward[a] = Mtrx.Vec(nStates, pomdp.useSparse);
			trueReward[a].set(pomdp.R[a]);
		}
		
		try {			
			cplex = new IloCplex();	
			cplex.setOut(null);
			
			// use Dual simplex algorithm
			//cplex.setParam(IloCplex.IntParam.RootAlg, IloCplex.Algorithm.Dual);
			// set Markowitz tolerance. default: 0.01 (0.0001 ~ 0.99999)
			cplex.setParam(IloCplex.DoubleParam.EpMrk, 0.99999);
			// set optimality tolerance. default: 1e-6 (1e-9 ~ 1e-1)
			cplex.setParam(IloCplex.DoubleParam.EpOpt, 1e-9);
			// set feasibility tolerance. default: 1e-6 (1e-9 ~ 1e-1)
			cplex.setParam(IloCplex.DoubleParam.EpRHS, 1e-9);
			
			optFscOccSA = calOccDist(optFsc);
//			for (int s = 0; s < nStates; s++) {
//				for (int a = 0; a < nActions; a++) {
//					if (optFscOccSA[s][a] != 0)
//						System.out.printf("occ[s%d][a%d] : %f\n", s, a, optFscOccSA[s][a]);
//				}
//			}
//			System.out.println();
		}
		catch (Exception ex) {
			System.err.println(ex);
		}
		
		// generate the set of sampled belief points for PBPI
		int maxRestart = 10;
		pbpiBeliefSet = BeliefPoints.initBeliefs(pomdp, nBeliefs, maxRestart, minDist, rand);
				
		// all possible combination of observation strategy
		System.out.printf("New policies : %f\n", 
				nNodes * nActions * Math.pow(nNodes, nObservs));
		etaList = IrlUtil.combination(nNodes, nObservs);	
	}
		
	public double solve(double lam) throws Exception {
		LAMBDA = lam;
		
		// number of variables
		int nR = nStates * nActions;
		int nEps = nNodes;
		int nOptV = nNodes * nStates;
		nVars = nR * 2 + nEps + nOptV;
		vars = new IloNumVar[nVars];
		varMap = new Hashtable<String, IloNumVar>();
				
		setVariables();
		setConstR();
		setConstOptV();
		setConstQ();
		setConstEps();
		setObjFun();

		nCols = cplex.getNcols();
		nRows = cplex.getNrows();
		System.out.printf("%d %d\n", nCols, nRows);
		cplex.exportModel("./output/QIrl_RockSample_4_4.lp");

		// solve lp
		double tmpT = CpuTime.getCurTime();
		if (cplex.solve()) {
			if (bPrint) {
				System.out.println("-- Result of CPLEX --");
				System.out.println("Objective value = " + cplex.getObjValue());
				System.out.println("Solution status = " + cplex.getStatus());
				for (int i = 0; i < vars.length; i++) 
					System.out.println(vars[i].getName() + " = " + cplex.getValue(vars[i]));
			}
		}
		else System.out.println("Solution status = " + cplex.getStatus());
		System.out.printf("# running time : %f \n\n", CpuTime.getElapsedTime(tmpT));
		
		//cplex.exportModel("./output/qirl.lp");
		//cplex.writeSolution("./output/qirl.lp");
		
		double obj = cplex.getObjValue();
		// save reward
		for (int a = 0; a < nActions; a++)
			for (int s = 0; s < nStates; s++) 
				learnedReward[s][a] = cplex.getValue(varMap.get(strR(s, a)));

		vars = null;
		varMap.clear();
		varMap = null;
		cplex.clearModel();
		return obj;		
	}
	
	public void setVariables() throws Exception {
		int id = 0;
		// R(s,a)
		for (int s = 0; s < nStates; s++) 
			for (int a = 0; a < nActions; a++) {
				String name = strR(s, a);
				vars[id] = cplex.numVar(R_MIN, R_MAX, name);
				varMap.put(name, vars[id++]);
			}
		
		// R2(s,a)
		for (int s = 0; s < nStates; s++) 
			for (int a = 0; a < nActions; a++) {
				String name = strR2(s, a);
				vars[id] = cplex.numVar(0, R_MAX, name);
				varMap.put(name, vars[id++]);
			}
		
		// Eps(n)
		for (int n = 0; n < nNodes; n++) {
			String name = strEps(n);
			vars[id] = cplex.numVar(0, INF, name);
			varMap.put(name, vars[id++]);
		}
		
		// optV(n,s)
		for (int n = 0; n < nNodes; n++)
			for (int s = 0; s < nStates; s++) {
				String name = strOptV(n, s);
				vars[id] = cplex.numVar(V_MIN, V_MAX, name);
				varMap.put(name, vars[id++]);
			}
	}
	
	public void setObjFun() throws Exception {
		// sum_n Eps(n) - lambda sum_s,a |R(s,a)|
		IloNumExpr exObj = cplex.numExpr();
		for (int n = 0; n < nNodes; n++)
			exObj = cplex.sum(exObj, varMap.get(strEps(n)));
        for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				IloNumExpr ex = cplex.prod(-LAMBDA, varMap.get(strR2(s, a)));
				exObj = cplex.sum(exObj, ex);
			}
		} 
        cplex.addMaximize(exObj);
	}
	
	public void setConstR() throws Exception {
		if (bPrint) System.out.println("-- Constraints for |R|");
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				// -R2(s,a) <= R(s,a)
				IloNumExpr exR2 = cplex.prod(-1, varMap.get(strR2(s, a)));
				cplex.addLe(exR2, varMap.get(strR(s,a)));				
				// R(s,a) <= R2(s,a)
				cplex.addLe(varMap.get(strR(s, a)), varMap.get(strR2(s, a)));				
			}
		}	
	}
	
	public void setConstOptV() throws Exception {
		if (bPrint) System.out.println("-- Contraints for value of optimal fsc");
		for (int n = 0; n < nNodes; n++) {
			for (int s = 0; s < nStates; s++) {
				// V(n,s) = R(s,psi(n)) 
				//        + gamma sum_s2 T(s,psi(n),s2) sum_z O(s2,psi(n),z) V(eta(n,z),s2) 
				int a = optFsc.getAction(n);
				IloNumExpr exL = varMap.get(strOptV(n, s));
				IloNumExpr exR = varMap.get(strR(s, a));
				
				Iterator<VectorEntry> itT = pomdp.T[a][s].iterator();
				while (itT.hasNext()) {
					VectorEntry veT = itT.next();
					int s2 = veT.index();
					
					Iterator<VectorEntry> itO = pomdp.O[a][s2].iterator();
					while (itO.hasNext()) {
						VectorEntry veO = itO.next();
						int z = veO.index();
						int n2 = optFsc.getNextNode(n, z);
						if (n2 != FscNode.NO_INFO) {
							double C = gamma * veT.get() * veO.get();
							IloNumVar varV2 = varMap.get(strOptV(n2, s2));
							exR = cplex.sum(exR, cplex.prod(C, varV2));
						}
						else 
							System.out.println("ERROR : undefined node transition in QIRL");
					}
				}
				cplex.addEq(exL, exR);
			}
		}
	}
	
	public void setConstEps() throws Exception {
		if (bPrint) 
			System.out.println("-- Contraints for sum_b,a,os Q(n,(psi_n,eta_n)) - Q(n,(a,os)) = eps(n)");
		for (int n = 0; n < nNodes; n++) {
			IloNumExpr exL = varMap.get(strEps(n));
			
			IloNumExpr exR = cplex.numExpr();
			int psi = optFsc.getAction(n);
			int[] eta = optFsc.getNode(n).nextNode;
			for (int a = 0; a < nActions; a++) {
				for (int zId = 0; zId < etaList.length; zId++) {
					int[] os = etaList[zId];
					for (Vector b : nodeBelief[n]) {
						exR = cplex.sum(exR, mkConstQ(n, psi, eta, a, os, b));
					}
				}
			}
			cplex.addEq(exL, exR);
		}		
	}
	
	public void setConstQ() throws Exception {
		if (bPrint) 
			System.out.println("-- Contraints for Q(n,(psi_n,eta_n)) - Q(n,(a,os)) >= 0");
		for (int n = 0; n < nNodes; n++) {	
			int psi = optFsc.getAction(n);
			int[] eta = optFsc.getNode(n).nextNode;
			for (int a = 0; a < nActions; a++) {
				for (int zId = 0; zId < etaList.length; zId++) {
					int[] os = etaList[zId];
					for (Vector b : nodeBelief[n]) {				
						IloNumExpr ex = mkConstQ(n, psi, eta, a, os, b);
						cplex.addGe(ex, 0);
					}
				}
			}
		}		
	}
	
	// sum_s b(s) * Q((n,s),(psi_n,eta_n)) - Q((n,s),(a,os))
	public IloNumExpr mkConstQ(int n, int psi, int[] eta, int a, int[] os, Vector B) 
	throws Exception {
		IloNumExpr ex = cplex.numExpr();		
		Iterator<VectorEntry> itB = B.iterator();
		while (itB.hasNext()) {
			VectorEntry veB = itB.next();
			int s = veB.index();
			double b = veB.get();
			
			// b(s) R(s,psi) - b(s) R(s,a)
			ex = cplex.sum(ex, cplex.prod(b, varMap.get(strR(s, psi))));
			ex = cplex.sum(ex, cplex.prod(-b, varMap.get(strR(s, a))));
			
			// b(s) gamma sum_{n2,s2} (T1 - T2)(s,n2,s2) V(n2,s2)
			Matrix T = mkT(s, psi, eta);
			T = T.add(-1, mkT(s, a, os));
			Iterator<MatrixEntry> itT = T.iterator();
			while (itT.hasNext()) {
				MatrixEntry meT = itT.next();
				int n2 = meT.row();
				int s2 = meT.column();
				double C = b * gamma * meT.get();
				ex = cplex.sum(ex, cplex.prod(C, varMap.get(strOptV(n2, s2))));
			}
			T = null;
		}
		return ex;
	}
	
	// T[(s,a,os),(n2,s2)] = T(s,a,s2) * sum_{z,os(z)=n2} O(a,s2,z)
	public Matrix mkT(int s, int a, int[] os) {
		Matrix T = Mtrx.Mat(nNodes, nStates, true);
		for (int n2 = 0; n2 < nNodes; n2++) {
			Iterator<VectorEntry> itT = pomdp.T[a][s].iterator();
			while (itT.hasNext()) {
				VectorEntry veT = itT.next();
				int s2 = veT.index();
				double pr = 0;
				for (int z = 0; z < nObservs; z++)
					if (os[z] == n2) 
						pr += pomdp.O[a][s2].get(z);
				if (pr != 0)
					T.set(n2, s2, veT.get() * pr);
			}
		}
		return T;
	}
	
	public String strR(int s, int a) { return String.format("R(s%d,a%d)", s, a); }
	public String strR2(int s, int a) { return String.format("R2(s%d,a%d)", s, a); }
	public String strEps(int n) { return String.format("eps(n%d)", n); }
	public String strOptV(int n, int s) { return String.format("V*(n%d,s%d)", n, s); }
	
	public double[][] getReward() {
		double[][] R = new double[nStates][nActions];
		for (int a = 0; a < nActions; a++) 
			for (int s = 0; s < nStates; s++) 
				R[s][a] = learnedReward[s][a];
		return R;
	}
	
	public double getSumR(double[][] R) {
		double sum = 0.0;
		for (int a = 0; a < nActions; a++) 
			for (int s = 0; s < nStates; s++) 
				sum += Math.abs(R[s][a]);
		return sum;
	}	
	
	public double[] eval(double[][] R, Random rand) {
		for (int a = 0; a < nActions; a++) {
			pomdp.R[a].zero();
			for (int s = 0; s < nStates; s++)
				if (R[s][a] != 0)
					pomdp.R[a].set(s, R[s][a]);
		}
		
		PBPI pbpi = new PBPI(pomdp, pbpiBeliefSet);
		pbpi.setParams(1000, 1e-6);
		FSC fsc = pbpi.run(false, rand);
		trueV = fsc.evaluation(trueReward);
		fscSize = fsc.size();
		//fsc.print();
		
		double V1 = optFsc.evaluation(pomdp.R);
		double V2 = fsc.evaluation(pomdp.R);
		double[] result = new double[3];
		result[0] = trueV;
		result[1] = V1;
		result[2] = V2;

		fsc.delete();
		fsc = null;
		pbpi.delete();
		pbpi = null;
		for (int a = 0; a < nActions; a++) {
			pomdp.R[a].zero();
			pomdp.R[a].set(trueReward[a]);
		}
		return result;
	}
	
	// reward shaping
	public double[][] transformReward(double[][] R) {
		FindShapingFunction shaping = 
			new FindShapingFunction(pomdp, R, false);
		transformedReward = shaping.find(optFscOccSA);
		return transformedReward;
	}
	
	public double calWeightedNorm(double[][] R, double p) throws Exception {
		double x = 0;
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				double y = Math.abs(trueReward[a].get(s) - R[s][a]) / pomdp.RMAX;
				x += optFscOccSA[s][a] * Math.pow(y, p);
			}
		}
		return Math.pow(x, 1.0 / p);
	}
	
	// calculate occupancy distribution using CPLEX
	public double[][] calOccDist(FSC fsc) throws Exception {
		Matrix occ = fsc.calOccDist();
				
		double[][] occSA = new double[nStates][nActions];
		Iterator<MatrixEntry> it = Mtrx.Iter(occ);
		while (it.hasNext()) {
			MatrixEntry me = it.next();
			int s = me.row();
			int n = me.column();
			occSA[s][fsc.getAction(n)] += me.get();
		}
//		for (int s = 0; s < nStates; s++) {
//			double sum = 0;
//			for (int a = 0; a < nActions; a++)
//				sum += occSA[s][a];
//			if (sum > 0) {
//				for (int a = 0; a < nActions; a++)
//					occSA[s][a] /= sum;
//			}
//		}
		return occSA;
	}

	public void printReward(double[][] R) {
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				if (R[s][a] != 0) {
					if (pomdp.states == null) System.out.printf("R[s%d]", s);
					else System.out.printf("R[%s]", pomdp.states[s]);
					if (pomdp.actions == null) System.out.printf("[a%d] : %.20f\n", a, R[s][a]);
					else System.out.printf("[%s] : %.20f\n", pomdp.actions[a], R[s][a]);
				}
			}
		}
		System.out.println();
	}
	
	public void delete() {
//		for (int z = 0; z < nObservs; z++) {
//			etaList[z].clear();
//			etaList[z] = null;
//		}
		etaList = null;
		learnedReward = null;
	}

}

