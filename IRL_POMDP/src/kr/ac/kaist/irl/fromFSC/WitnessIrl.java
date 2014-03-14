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

import kr.ac.kaist.pomdp.data.BeliefPoints;
import kr.ac.kaist.pomdp.data.FSC;
import kr.ac.kaist.pomdp.data.FscNode;
import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.pomdp.pbpi.PBPI;
import kr.ac.kaist.utils.IrlUtil;
import kr.ac.kaist.utils.Mtrx;

/**
 * Solve inverse reinforcement learning (IRL) problems 
 * for partially observable environments using the witness theorem based optimality constraint
 * when the expert policy is explicitly given in the form of a finite state controller (FSC).
 *
 * (N', pi') = dp-backup(N, pi) using the witness thm.
 * sum_s b(n,s) V^pi(n,s) >= sum_s b(n,s) V^pi'(n',s)
 * sum_s b(n,s) V^pi(n,s) - sum_s b(n,s) V^pi'(n',s) = eps(n,b)
 * for all n in N, all n' in N', all b in B(n)
 * obj: sum eps(n,b) - lambda*|R|
 *
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, IJCAI 2009.
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, JMLR 12, 2011.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class WitnessIrl {
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
	private ArrayList<Vector>[] nodeBelief;
	private ArrayList<FscNode> dpList;
	private double[][] optFscOccSA;

	private IloCplex cplex;
	private int nVars;
	private IloNumVar[] vars;
	private Hashtable<String, IloNumVar> varMap;
	public int nCols;
	public int nRows;
	
	private Vector[] trueReward;
	private double[][] learnedReward;
	private boolean bPrint;

	// variables of PBPI
	private ArrayList<Vector> pbpiBeliefSet;
	public double trueV;
	public int fscSize;
	
	public WitnessIrl(PomdpProblem _pomdp, FSC _fsc, 
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
		useSparse = pomdp.useSparse;
		V_MAX = R_MAX / (1 - gamma);
		V_MIN = R_MIN / (1 - gamma);
		learnedReward = new double[nStates][nActions];
		trueReward = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			trueReward[a] = Mtrx.Vec(nStates, useSparse);
			trueReward[a].set(pomdp.R[a]);
		}
		
		try {
			cplex = new IloCplex();	
			cplex.setOut(null);
			
			// use Dual simplex algorithm. default: auto
			//cplex.setParam(IloCplex.IntParam.RootAlg, IloCplex.Algorithm.Dual);
			// set Markowitz tolerance. default: 0.01 (0.0001 ~ 0.99999)
			cplex.setParam(IloCplex.DoubleParam.EpMrk, 0.99999);
			// set optimality tolerance. default: 1e-6 (1e-9 ~ 1e-1)
			cplex.setParam(IloCplex.DoubleParam.EpOpt, 1e-9);
			// set feasibility tolerance. default: 1e-6 (1e-9 ~ 1e-1)
			cplex.setParam(IloCplex.DoubleParam.EpRHS, 1e-9);
			// emphasize precision in numerically unstable or difficult problems. default: false
			//cplex.setParam(IloCplex.BooleanParam.NumericalEmphasis, true);
						
			optFscOccSA = IrlUtil.calOccSA(pomdp, optFsc);
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
		System.out.printf("+ # of beliefs for PBPI : %d\n", pbpiBeliefSet.size());
		
		// generate new nodes that will be compared with the given expert's FSC
		System.out.printf("New DP nodes            : %d", 
				nNodes * nNodes * nObservs);
		dpList = optFsc.wBackup();
		nDpNodes = dpList.size();
		System.out.printf(", %d\n", nDpNodes);
	}
	
	public double solve(double lam) throws Exception {
		LAMBDA = lam;				
		
		// number of variables
		int nR = nStates * nActions;
		int nOptV = nNodes * nStates;
		int nDpV = nDpNodes * nStates;
		nVars = nR * 2 + nOptV + nDpV;
		vars = new IloNumVar[nVars];
		varMap = new Hashtable<String, IloNumVar>();
				
		setVariables();
		setConstR();
		setConstOptV();
		setConstDpV();
		setConstIneq();
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
		else {
			System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
			System.out.println("Objective value = " + cplex.getObjValue());
			System.out.println("Solution status = " + cplex.getStatus());
			/*int i = 0;
			for (int a = 0; a < nActions; a++) {
				for (int s = 0; s < nStates; s++) {
					System.out.printf("%s = %.20f (%f)\n", 
							vars[i].getName(), cplex.getValue(vars[i]), reward[s][a]);
					i++;
				}
			}*/
			System.out.println("Is primal feasible? " + cplex.isPrimalFeasible());
			System.out.println("Is dual feasible? " + cplex.isDualFeasible());
			
			IloCplex.Quality inf = cplex.getQuality(IloCplex.QualityType.MaxPrimalInfeas);
			double maxinfeas = inf.getValue();
			System.out.printf("Solution quality : %.20f\n", maxinfeas);
		}

		double obj = cplex.getObjValue();
		// save reward
		for (int a = 0; a < nActions; a++)
			for (int s = 0; s < nStates; s++) 
				learnedReward[s][a] = cplex.getValue(varMap.get(strR(s, a)));		
				
		nCols = cplex.getNcols();
		nRows = cplex.getNrows();
		vars = null;
		varMap.clear();
		varMap = null;
		cplex.clearModel();
		return obj;
	}
	
	private void setVariables() throws Exception {
		int id = 0;
		// R(s,a)
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				String name = strR(s, a);
				vars[id] = cplex.numVar(R_MIN, R_MAX, name);
				varMap.put(name, vars[id++]);
			}
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
	
	private void setObjFun() throws Exception {
		// sum_n sum_b sum_n2 Eps(n,b,n2) - lambda sum_s,a |R(s,a)|
		IloNumExpr exObj = cplex.numExpr();
		for (int n = 0; n < nNodes; n++) {	
			for (int n2 = 0; n2 < nDpNodes; n2++) {
				for (Vector B : nodeBelief[n]) {
					Iterator<VectorEntry> itB = Mtrx.Iter(B);
					while (itB.hasNext()) {
						VectorEntry veB = itB.next();
						int s = veB.index();
						double pr = veB.get(); // / nodeBelief[n].size();		
						exObj = cplex.sum(exObj, cplex.prod(pr, varMap.get(strOptV(n, s))));
						exObj = cplex.sum(exObj, cplex.prod(-pr, varMap.get(strDpV(n2, s))));
					}
				}
			}
		}
		for (int s = 0; s < nStates; s++)
			for (int a = 0; a < nActions; a++)
				exObj = cplex.sum(exObj, cplex.prod(-LAMBDA, varMap.get(strR2(s, a))));
		cplex.addMaximize(exObj);
	}

	private void setConstR() throws Exception {		
		if (bPrint) System.out.println("-- Constraints for |R|");
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				// -R2(s,a) <= R(s,a)
				cplex.addLe(cplex.prod(-1, varMap.get(strR2(s, a))), varMap.get(strR(s,a)));				
				// R(s,a) <= R2(s,a)
				cplex.addLe(varMap.get(strR(s, a)), varMap.get(strR2(s, a)));
			}
		}		
	}
	
	private void setConstOptV() throws Exception {
		if (bPrint) System.out.println("-- value of optimal fsc");		
		for (int n = 0; n < nNodes; n++) {
			int a = optFsc.getAction(n);
			for (int s = 0; s < nStates; s++) { 
				// V(n,s) = R(s,psi(n)) 
				//        + gamma sum_s2 T(s,psi(n),s2) sum_z O(s2,psi(n),z) V(eta(n,z),s2)
				IloNumExpr exL = varMap.get(strOptV(n, s));
				IloNumExpr exR = varMap.get(strR(s, a));
				
				Iterator<VectorEntry> itT = Mtrx.Iter(pomdp.T[a][s]);
				while (itT.hasNext()) {
					VectorEntry veT = itT.next();
					int s2 = veT.index();
					
					Iterator<VectorEntry> itO = Mtrx.Iter(pomdp.O[a][s2]);
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
	
	private void setConstDpV() throws Exception {
		if (bPrint) System.out.println("-- value of newly generated nodes");
		for (int n = 0; n < nDpNodes; n++) {
			FscNode node = dpList.get(n);
			int a = node.act;
			for (int s = 0; s < nStates; s++) {
				// V'(n,s) = R(s,psi'(n)) 
				//         + gamma sum_s2 T(s,psi'(n),s2) sum_z O(s2,psi'(n),z) V(eta'(n,z),s2)
				IloNumExpr exL = varMap.get(strDpV(n, s));
				IloNumExpr exR = varMap.get(strR(s, a));
				
				Iterator<VectorEntry> itT = Mtrx.Iter(pomdp.T[a][s]);
				while (itT.hasNext()) {
					VectorEntry veT = itT.next();
					int s2 = veT.index();
					
					Iterator<VectorEntry> itO = Mtrx.Iter(pomdp.O[a][s2]);
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
	
	private void setConstIneq() throws Exception {
		if (bPrint) System.out.println("-- Contraints for b * V_opt >= b * V_dp");
		for (int n = 0; n < nNodes; n++) {
			for (int n2 = 0; n2 < dpList.size(); n2++) {
				for (Vector B : nodeBelief[n]) {					
					// sum_s b(s) V(n,s) >= sum_s b(s) V'(n2,s)
					IloNumExpr exL = cplex.numExpr();
					IloNumExpr exR = cplex.numExpr();
					Iterator<VectorEntry> itB = B.iterator();
					while (itB.hasNext()) {
						VectorEntry veB = itB.next();
						int s = veB.index();
						double b = veB.get();
						exL = cplex.sum(exL, cplex.prod(b, varMap.get(strOptV(n, s))));
						exR = cplex.sum(exR, cplex.prod(b, varMap.get(strDpV(n2, s))));
					}
					cplex.addGe(exL, exR);
				}
			}
		}
	}
	
	private String strR(int s, int a) { return String.format("R(s%d,a%d)", s, a); }
	private String strR2(int s, int a) { return String.format("R2(s%d,a%d)", s, a); }
	private String strOptV(int n, int s) { return String.format("V*(n%d,s%d)", n, s); }
	private String strDpV(int n, int s) { return String.format("V'(n%d,s%d)", n, s); }
	
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
				if (R[s][a] != 0) pomdp.R[a].set(s, R[s][a]);
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
	
	public double calWeightedNorm(double[][] R, double p) throws Exception {
		Vector[] R2 = new Vector[nActions];
		for (int a = 0; a < nActions; a++) {
			R2[a] = Mtrx.Vec(nStates, useSparse);
			for (int s = 0; s < nStates; s++) {
				if (R[s][a] != 0) R2[a].set(s, R[s][a]);
			}
		}
		return IrlUtil.calWeightedNorm(pomdp, trueReward, R2, optFscOccSA);
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

