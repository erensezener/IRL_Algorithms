package kr.ac.kaist.irl.fromFSC;

import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Random;

import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;

import kr.ac.kaist.pomdp.data.*;
import kr.ac.kaist.pomdp.pbpi.PBPI;
import kr.ac.kaist.utils.IrlUtil;
import kr.ac.kaist.utils.Mtrx;

/**
 * Solve inverse reinforcement learning (IRL) problems 
 * for partially observable environments using the witness theorem based optimality constraint
 * when the expert policy is explicitly given in the form of a finite state controller (FSC).
 * 
 * Use basis functions and compute the weight of basis functions.
 *
 * (N', pi') = dp-backup(N, pi) using Witness Thm.
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
public class WitnessIrl2 {
	private final double INF = 1e5;
	private final double R_MAX = 1;	
	private final double R_MIN = -1;
	private double LAMBDA;
	private double V_MAX;
	private double V_MIN;

	private PomdpProblem pomdp;		
	private int nStates;
	private int nActions;
	private int nObservs;
	private double gamma;
	private boolean useSparse;

	private int nNodes;
	private int nDpNodes;
	private FSC optFsc;
	private ArrayList[] nodeBelief;
	private ArrayList<FscNode> dpList;

	private IloCplex cplex;
	private int nVars;
	private IloNumVar[] vars;
	private Hashtable<String, IloNumVar> varMap;
	public int nCols;
	public int nRows;
	
	private int phiNum;
	private BasisFunctions phi;
	private Vector W;
	private Vector[] trueReward;
	private double[][] learnedReward;
	private boolean bPrint;

	// variables of PBPI
	private ArrayList<Vector> pbpiBeliefSet;
	public double trueV;
	public int fscSize;
	
	public WitnessIrl2(PomdpProblem _pomdp, FSC _fsc, BasisFunctions _phi,
			int nBeliefs, double minDist, boolean _bPrint, Random rand) {
		pomdp = _pomdp;
		optFsc = _fsc;
		nodeBelief = _fsc.getNodeBelief();
		bPrint = _bPrint;
		phi = _phi;
		phiNum = phi.getNBasis();
		
		nNodes = optFsc.size();
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		nObservs = pomdp.nObservations;
		gamma = pomdp.gamma;
		useSparse = pomdp.useSparse;
		V_MAX = R_MAX / (1 - gamma);
		V_MIN = R_MIN / (1 - gamma);
		
		W = Mtrx.Vec(phiNum, useSparse);
		learnedReward = new double[nStates][nActions];
		trueReward = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			trueReward[a] = Mtrx.Vec(nStates, useSparse);
			trueReward[a].set(pomdp.R[a]);
		}
		
		try {
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
			// emphasize precision in numerically unstable or difficult problems. default: false
			cplex.setParam(IloCplex.BooleanParam.NumericalEmphasis, true);
		}
		catch (Exception ex) {
			System.err.println(ex);
		}
		
		// generate the set of sampled belief points for PBPI
		int maxRestart = 10;
		pbpiBeliefSet = BeliefPoints.initBeliefs(pomdp, nBeliefs, maxRestart, minDist, rand);
		System.out.printf("+ # of beliefs for PBPI : %d\n", pbpiBeliefSet.size());
		
		// generate new nodes that will be compared with the given expert's FSC
		System.out.printf("+ New DP nodes          : %d", 
				nNodes * nNodes * nObservs);
		dpList = optFsc.wBackup();
		nDpNodes = dpList.size();
		System.out.printf(", %d\n", nDpNodes);
		System.out.printf("+ # of basis function   : %d\n", phi.getNBasis());
	}
	
	public double solve(double lam) throws Exception {
		LAMBDA = lam;
		
		// number of variables
		int nW = phiNum;
		int nR2 = nStates * nActions;
		int nOptV = nNodes * nStates;
		int nDpV = nDpNodes * nStates;
		nVars = nW + nR2 + nOptV + nDpV;
		vars = new IloNumVar[nVars];
		varMap = new Hashtable<String, IloNumVar>();
				
		setVariables();
		setConstOptV();
		setConstDpV();
		setConstIneq();
		setConstR();
		setObjFun();
		
		// solve lp
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

		double obj = cplex.getObjValue();
		W.zero();
		for (int p = 0; p < phiNum; p++) {
			double x = cplex.getValue(varMap.get(strW(p)));
			if (x != 0.0) W.set(p, x);
		}
			
		// save reward
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				learnedReward[s][a] = 0;
				for (int p = 0; p < phiNum; p++)
					learnedReward[s][a] += W.get(p) * phi.get(p, s, a);
			}
		}
		
		nCols = cplex.getNcols();
		nRows = cplex.getNrows();
		vars = null;
		varMap.clear();
		varMap = null;
		cplex.clearModel();
		return obj;		
	}
	
	public void setVariables() throws Exception {
		int id = 0;
		
		// W
		for (int p = 0; p < phiNum; p++) {
			String name = strW(p);
			vars[id] = cplex.numVar(-1, 1, name);
			varMap.put(name, vars[id++]);
		}
		
		// R2(s,a)
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				String name = strR2(s, a);
				vars[id] = cplex.numVar(0, R_MAX, name);
				varMap.put(name, vars[id++]);
			}
		}
		
		// optV(n,s)
		for (int n = 0; n < nNodes; n++) {
			for (int s = 0; s < nStates; s++) {
				String name = strOptV(n, s);
				vars[id] = cplex.numVar(V_MIN, V_MAX, name);
				varMap.put(name, vars[id++]);
			}
		}
		
		// dpV(n,s)
		for (int n = 0; n < nDpNodes; n++) {
			for (int s = 0; s < nStates; s++)  {
				String name = strDpV(n, s);
				vars[id] = cplex.numVar(V_MIN, V_MAX, name);
				varMap.put(name, vars[id++]);
			}		
		}
	}
	
	public void setObjFun() throws Exception {
		// sum_n sum_b sum_n2 sum_s [b(s) V(n,s) - b(s) V'(n2,s)] - lambda * |W|_1
		IloNumExpr exObj = cplex.numExpr();
		for (int n = 0; n < nNodes; n++) {	
			for (int b = 0; b < nodeBelief[n].size(); b++) {
				Vector B = (Vector) nodeBelief[n].get(b);
				for (int n2 = 0; n2 < nDpNodes; n2++) {
					Iterator<VectorEntry> itB = B.iterator();
					while (itB.hasNext()) {
						VectorEntry veB = itB.next();
						int s = veB.index();
						double pr = veB.get();						
						exObj = cplex.sum(exObj, cplex.prod(pr, varMap.get(strOptV(n, s))));
						exObj = cplex.sum(exObj, cplex.prod(-pr, varMap.get(strDpV(n2, s))));
					}
				}
			}
		}
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
		// -R2(s,a) <= sum_p alpha(p) * phi(p,s,a) <= R2(s,a)
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				IloNumExpr exp1L = cplex.prod(-1, varMap.get(strR2(s, a)));
				IloNumExpr exp1R = cplex.numExpr();
				for (int p = 0; p < phiNum; p++) {
					IloNumVar varA = varMap.get(strW(p));
					exp1R = cplex.sum(exp1R, cplex.prod(phi.get(p, s, a), varA));
				}
				cplex.addLe(exp1L, exp1R);

				IloNumExpr exp2L = cplex.numExpr();
				for (int p = 0; p < phiNum; p++) {
					IloNumVar varA = varMap.get(strW(p));
					exp2L = cplex.sum(exp2L, cplex.prod(phi.get(p, s, a), varA));
				}
				cplex.addLe(exp2L, varMap.get(strR2(s, a)));
			}
		}	
	}
	
	public void setConstOptV() throws Exception {
		// V(n,s) = sum_p W(p) * phi(p, s, a) 
		//        + gamma sum_s2 T(s,psi(n),s2) sum_z O(s2,psi(n),z) V(eta(n,z),s2)
		if (bPrint) System.out.println("-- value of optimal fsc");		
		for (int n = 0; n < nNodes; n++) { 
			int a = optFsc.getAction(n);
			for (int s = 0; s < nStates; s++) {
				IloNumExpr exL = varMap.get(strOptV(n, s));
				
				IloNumExpr exR = cplex.numExpr();
				for (int p = 0; p < phiNum; p++)
					exR = cplex.sum(exR, cplex.prod(phi.get(p, s, a), varMap.get(strW(p))));
				
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
					}
				}
				cplex.addEq(exL, exR);
			}
		}
	}
	
	public void setConstDpV() throws Exception {
		// V'(n,s) = sum_p W(p) * phi(p, s, a) 
		//         + gamma sum_s2 T(s,psi'(n),s2) sum_z O(s2,psi'(n),z) V(eta'(n,z),s2)
		if (bPrint) System.out.println("-- value of dp-backuped nodes");
		for (int n = 0; n < nDpNodes; n++) {
			FscNode node = dpList.get(n);
			int a = node.act;
			for (int s = 0; s < nStates; s++) {
				IloNumExpr exL = varMap.get(strDpV(n, s));

				IloNumExpr exR = cplex.numExpr();
				for (int p = 0; p < phiNum; p++)
					exR = cplex.sum(exR, cplex.prod(phi.get(p, s, a), varMap.get(strW(p))));
				
				Iterator<VectorEntry> itT = pomdp.T[a][s].iterator();
				while (itT.hasNext()) {
					VectorEntry veT = itT.next();
					int s2 = veT.index();
					
					Iterator<VectorEntry> itO = pomdp.O[a][s2].iterator();
					while (itO.hasNext()) {
						VectorEntry veO = itO.next();
						int z = veO.index();
						int n2 = node.nextNode[z];
						if (n2 != FscNode.NO_INFO) {
							double C = gamma * veT.get() * veO.get();
							IloNumVar varV2 = varMap.get(strOptV(n2, s2));
							exR = cplex.sum(exR, cplex.prod(C, varV2));
						}
					}
				}
				cplex.addEq(exL, exR);
			}
		}
	}
	
	public void setConstIneq() throws Exception {
		if (bPrint) System.out.println("-- sum_s b(s) V(n,s) >= sum_s b(s) V'(n2,s)");
		for (int n = 0; n < nNodes; n++) {			
			for (int n2 = 0; n2 < nDpNodes; n2++) {
				for (int b = 0; b < nodeBelief[n].size(); b++) {
					IloNumExpr exL = cplex.numExpr();
					IloNumExpr exR = cplex.numExpr();
					Vector B = (Vector) nodeBelief[n].get(b);
					Iterator<VectorEntry> itB = B.iterator();
					while (itB.hasNext()) {
						VectorEntry veB2 = itB.next();
						int s = veB2.index();
						double pr = veB2.get();
						exL = cplex.sum(exL, cplex.prod(pr, varMap.get(strOptV(n, s))));
						exR = cplex.sum(exR, cplex.prod(pr, varMap.get(strDpV(n2, s))));
					}
					cplex.addGe(exL, exR);
				}
			}
		}
	}
	
	public String strW(int p) { return String.format("Phi(%d)", p); }
	public String strR2(int s, int a) { return String.format("R2(s%d,a%d)", s, a); }
	public String strOptV(int n, int s) { return String.format("V*(n%d,s%d)", n, s); }
	public String strDpV(int n, int s) { return String.format("V'(n%d,s%d)", n, s); }
	
	public double[] getW() {
		double[] weight = new double[phiNum];
		for (int p = 0; p < phiNum; p++)
			weight[p] = W.get(p);
		return weight;
	}
	
	public double getSumW() {
		double sum = 0.0;
		for (int p = 0; p < phiNum; p++)
			sum += Math.abs(W.get(p));
		return sum;
	}
	
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
				if (learnedReward[s][a] != 0.0)
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
	
	public void printW() {
		System.out.println("Weight of basis function");
		Iterator<VectorEntry> itW = Mtrx.Iter(W);
		while (itW.hasNext()) {
			VectorEntry veW = itW.next();
			System.out.printf("  %d: %.15f\n", veW.index(), veW.get());
		}
		System.out.println();
	}
	
	public void printReward(double[][] R) {
		for (int a = 0; a < nActions; a++) {
			if (pomdp.actions == null) System.out.printf("Reward: %d\n", a);
			else System.out.printf("Reward: %s\n", pomdp.actions[a]);
			for (int s = 0; s < nStates; s++)
				if (R[s][a] != 0)
					if (pomdp.states == null) 
						System.out.printf("  %d: %.15f\n", s, R[s][a]);
					else 
						System.out.printf("  %s: %.15f\n", pomdp.states[s], R[s][a]);
			System.out.println();
		}		
	}

	public void printDpNodes() { 
		IrlUtil.print(pomdp, dpList); 
	}

	public void delete() {
		if (cplex != null) {
			cplex.end();	
			cplex = null;
		}
		dpList.clear();
		dpList = null;
		learnedReward = null;
	}

}

