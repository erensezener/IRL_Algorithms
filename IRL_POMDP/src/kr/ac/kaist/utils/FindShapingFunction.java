package kr.ac.kaist.utils;

import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.Enumeration;
import java.util.Hashtable;

import no.uib.cipr.matrix.Vector;

import kr.ac.kaist.pomdp.data.PomdpProblem;

public class FindShapingFunction {
	final private double INF = 1e5;
	
	private PomdpProblem pomdp;		
	private int nStates;
	private int nActions;
	private double gamma;
	
	private IloCplex cplex;
	private Hashtable<String, IloNumVar> varMap;
	
	private double[][] trueR;
	private double[][] learnedR;
	private double[][] transformedR;
	private double rmax;
	private boolean bPrint;
	
	public FindShapingFunction(PomdpProblem _pomdp, double[][] R, boolean _bPrint) {
		pomdp = _pomdp;
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		gamma = pomdp.gamma;
		bPrint = _bPrint;
		
		trueR = new double[nStates][nActions];
		learnedR = new double[nStates][nActions];
		rmax = Double.NEGATIVE_INFINITY;
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				trueR[s][a] = pomdp.R[a].get(s);
				learnedR[s][a] = R[s][a];
				if (Math.abs(trueR[s][a]) > rmax) rmax = Math.abs(trueR[s][a]);
			}
		}
		
		if (bPrint) {
			System.out.println("True reward function");
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					System.out.printf("  s%2d, a%2d : %25.20f\n", s, a, trueR[s][a]);
				}
			}
			System.out.println();
			
			System.out.println("Learned reward function");
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					System.out.printf("  s%2d, a%2d : %25.20f\n", s, a, learnedR[s][a]);
				}
			}
			System.out.println();
		}
	}
	
	public FindShapingFunction(PomdpProblem _pomdp, Vector[] R, boolean _bPrint) {
		pomdp = _pomdp;
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		gamma = pomdp.gamma;
		bPrint = _bPrint;
		
		trueR = new double[nStates][nActions];
		learnedR = new double[nStates][nActions];
		rmax = Double.NEGATIVE_INFINITY;
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				trueR[s][a] = pomdp.R[a].get(s);
				learnedR[s][a] = R[a].get(s);
				if (Math.abs(trueR[s][a]) > rmax) rmax = Math.abs(trueR[s][a]);
			}
		}
		
		if (bPrint) {
			System.out.println("True reward function");
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					System.out.printf("  s%2d, a%2d : %25.20f\n", s, a, trueR[s][a]);
				}
			}
			System.out.println();
			
			System.out.println("Learned reward function");
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					System.out.printf("  s%2d, a%2d : %25.20f\n", s, a, learnedR[s][a]);
				}
			}
			System.out.println();
		}
	}
	
	public double[][] find(double[][] occ) {
		try {
			cplex = new IloCplex();	
			cplex.setOut(null);
			
			// use Dual simplex algorithm
			cplex.setParam(IloCplex.IntParam.RootAlg, IloCplex.Algorithm.Dual);
			// set Markowitz tolerance. default: 0.01 (0.0001 ~ 0.99999)
			cplex.setParam(IloCplex.DoubleParam.EpMrk, 0.99999);
			// set optimality tolerance. default: 1e-6 (1e-9 ~ 1e-1)
			cplex.setParam(IloCplex.DoubleParam.EpOpt, 1e-9);
			// set feasibility tolerance. default: 1e-6 (1e-9 ~ 1e-1)
			cplex.setParam(IloCplex.DoubleParam.EpRHS, 1e-9);
			// emphasize precision in numerically unstable or difficult problems. default: false
			cplex.setParam(IloCplex.BooleanParam.NumericalEmphasis, true);
			
			varMap = new Hashtable<String, IloNumVar>();			
			for (int s = 0; s < nStates; s++) {
				String name = strPhi(s);
				varMap.put(name, cplex.numVar(-INF, INF, name));
			}

			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					String name = strR2(s, a);
					varMap.put(name, cplex.numVar(-rmax, rmax, name));
				}
			}
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					String name = strEps(s, a);
					varMap.put(name, cplex.numVar(0, rmax, name));
				}
			}
			varMap.put("C", cplex.numVar(-INF, INF, "C"));
			
			// R2(s,a) = C * learnedR(s,a) + sum_s' T(s,a,s') [gamma Phi(s') - Phi(s)]
			//         = C * learnedR(s,a) - Phi(s) + sum_s' gamma T(s,a,s') Phi(s')
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					IloNumExpr expR2 = cplex.prod(learnedR[s][a], varMap.get("C"));
					expR2 = cplex.sum(expR2, cplex.prod(-1, varMap.get(strPhi(s))));
					for (int s2 = 0; s2 < nStates; s2++) {
						double c = gamma * pomdp.T[a][s].get(s2);
						expR2 = cplex.sum(expR2, cplex.prod(c, varMap.get(strPhi(s2))));
					}
					cplex.addEq(varMap.get(strR2(s, a)), expR2);
				}
			}
			
			// -eps(s,a) <= trueR(s,a) - R2(s,a) <= eps(s,a)
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					IloNumExpr ex1 = cplex.sum(trueR[s][a], cplex.prod(-1, varMap.get(strR2(s,a))));
					cplex.addLe(cplex.prod(-1, varMap.get(strEps(s, a))), ex1);
					
					IloNumExpr ex2 = cplex.sum(trueR[s][a], cplex.prod(-1, varMap.get(strR2(s,a))));
					cplex.addLe(ex2, varMap.get(strEps(s, a)));
				}
			}
			
			// minimize sum_s,a eps(s,a)
			IloNumExpr expObj = cplex.numExpr();
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
//					IloNumExpr ex = cplex.prod(occ[s][a], varMap.get(strEps(s, a)));
//					expObj = cplex.sum(expObj, ex);
					expObj = cplex.sum(expObj, varMap.get(strEps(s, a)));
				}
			}
			cplex.addMinimize(expObj);
			
			// solve lp
			if (cplex.solve()) {
				if (bPrint) {
//					System.out.println("-- Result of CPLEX --");
//					System.out.println("Objective value = " + cplex.getObjValue());
//					System.out.println("Solution status = " + cplex.getStatus());
//					Enumeration<String> keys = varMap.keys();
//					Enumeration<IloNumVar> elements = varMap.elements();
//					while (keys.hasMoreElements()) {
//						System.out.println(keys.nextElement() + " = " + cplex.getValue(elements.nextElement()));
//					}
//					System.out.println();
					
					System.out.println("Transformed reward function");
					for (int s = 0; s < nStates; s++) {
						for (int a = 0; a < nActions; a++) {
							double r = cplex.getValue(varMap.get(strR2(s, a)));
							System.out.printf("  s%2d, a%2d : %25.20f\n", s, a, r);
						}
					}
					System.out.println();
				}
			}
			
			transformedR = new double[nStates][nActions];
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					transformedR[s][a] = cplex.getValue(varMap.get(strR2(s, a)));
				}
			}

			varMap.clear();
			cplex.clearModel();
		}
		catch (Exception ex) {
			System.err.println(ex);
		}
		
		return transformedR;
	}
	
	public double[][] find(double[][] occ, double S) {
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
			// emphasize precision in numerically unstable or difficult problems. default: false
			cplex.setParam(IloCplex.BooleanParam.NumericalEmphasis, true);
			
			varMap = new Hashtable<String, IloNumVar>();			
			for (int s = 0; s < nStates; s++) {
				String name = strPhi(s);
				varMap.put(name, cplex.numVar(-INF, INF, name));
			}

			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					String name = strR2(s, a);
					varMap.put(name, cplex.numVar(-rmax, rmax, name));
				}
			}
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					String name = strEps(s, a);
					varMap.put(name, cplex.numVar(0, rmax, name));
				}
			}
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					String name = strX(s, a);
					varMap.put(name, cplex.intVar(0, 1, name));
				}
			}
			varMap.put("C", cplex.numVar(-INF, INF, "C"));
			
			// R2(s,a) = C * learnedR(s,a) + sum_s' T(s,a,s') [gamma Phi(s') - Phi(s)]
			//         = C * learnedR(s,a) - Phi(s) + sum_s' gamma T(s,a,s') Phi(s')
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					IloNumExpr expR2 = cplex.prod(learnedR[s][a], varMap.get("C"));
					expR2 = cplex.sum(expR2, cplex.prod(-1, varMap.get(strPhi(s))));
					for (int s2 = 0; s2 < nStates; s2++) {
						double c = gamma * pomdp.T[a][s].get(s2);
						expR2 = cplex.sum(expR2, cplex.prod(c, varMap.get(strPhi(s2))));
					}
					cplex.addEq(varMap.get(strR2(s, a)), expR2);
				}
			}
			
			// -eps(s,a) <= R2(s,a) <= eps(s,a)
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					cplex.addLe(cplex.prod(-1, varMap.get(strEps(s, a))), varMap.get(strR2(s,a)));
					cplex.addLe(varMap.get(strR2(s,a)), varMap.get(strEps(s, a)));
				}
			}
			
			// sum_s,a eps(s,a) >= S
			IloNumExpr exSum = cplex.numExpr();
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					exSum = cplex.sum(exSum, varMap.get(strEps(s, a)));
				}
			}
			cplex.addEq(exSum, S);
			
			// eps(s,a) <= M * X(s,a)
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					IloNumExpr exp = cplex.prod(rmax, varMap.get(strX(s, a)));
					cplex.addLe(varMap.get(strEps(s, a)), exp);
				}
			}
			
			// minimize sum_s,a X(s,a)
			IloNumExpr expObj = cplex.numExpr();
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					expObj = cplex.sum(expObj, varMap.get(strX(s, a)));
				}
			}
			cplex.addMinimize(expObj);
			
			// solve lp
			if (cplex.solve()) {
				if (bPrint) {
					System.out.println("-- Result of CPLEX --");
					System.out.println("Objective value = " + cplex.getObjValue());
					System.out.println("Solution status = " + cplex.getStatus());
					Enumeration<String> keys = varMap.keys();
					Enumeration<IloNumVar> elements = varMap.elements();
					while (keys.hasMoreElements()) {
						String x = keys.nextElement().toString();
						double y = (double) cplex.getValue(elements.nextElement());
						if (y > 0)
							System.out.println(x + " = " + y);
					}
					System.out.println();
					
					System.out.println("Transformed reward function");
					for (int s = 0; s < nStates; s++) {
						for (int a = 0; a < nActions; a++) {
							double r = cplex.getValue(varMap.get(strR2(s, a)));
							if (r > 0)
								System.out.printf("  s%2d, a%2d : %25.20f\n", s, a, r);
						}
					}
					System.out.println();
				}
			}
			
			transformedR = new double[nStates][nActions];
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					transformedR[s][a] = cplex.getValue(varMap.get(strR2(s, a)));
				}
			}

			varMap.clear();
			cplex.clearModel();
		}
		catch (Exception ex) {
			System.err.println(ex);
		}
		
		return transformedR;
	}
	
	public double[][] find2(double[][] occ) {
		try {
			cplex = new IloCplex();	
			cplex.setOut(null);
						
			varMap = new Hashtable<String, IloNumVar>();
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					String name = strR2(s, a);
					varMap.put(name, cplex.numVar(-rmax, rmax, name));
				}
			}
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					String name = strEps(s, a);
					varMap.put(name, cplex.numVar(0, rmax * 2, name));
				}
			}
			varMap.put("C1", cplex.numVar(-INF, INF, "C1"));
			varMap.put("C2", cplex.numVar(-INF, INF, "C2"));
			
			// R2(s,a) = C1 * learnedR(s,a) + C2;
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					IloNumExpr expR2 = cplex.prod(learnedR[s][a], varMap.get("C1"));
					expR2 = cplex.sum(expR2, varMap.get("C2"));
					cplex.addEq(varMap.get(strR2(s, a)), expR2);
				}
			}
			
			// -eps(s,a) <= trueR(s,a) - R2(s,a) <= eps(s,a)
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					IloNumExpr eps = cplex.prod(-1, varMap.get(strEps(s, a)));
					IloNumExpr expEps = cplex.prod(-1, varMap.get(strR2(s,a)));
					expEps = cplex.sum(expEps, trueR[s][a]);
					cplex.addLe(eps, expEps);
					
					IloNumExpr expEps2 = cplex.prod(-1, varMap.get(strR2(s,a)));
					expEps2 = cplex.sum(expEps2, trueR[s][a]);
					cplex.addLe(expEps2, varMap.get(strEps(s, a)));
				}
			}
			
			// minimize sum_s,a eps(s,a)
			IloNumExpr expObj = cplex.numExpr();
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
//					IloNumExpr ex = cplex.prod(occ[s][a], varMap.get(strEps(s, a)));
//					expObj = cplex.sum(expObj, ex);
					expObj = cplex.sum(expObj, varMap.get(strEps(s, a)));
				}
			}
			cplex.addMinimize(expObj);
			
			// solve lp
			if (cplex.solve()) {
				if (bPrint) {
					System.out.println("-- Result of CPLEX --");
					System.out.println("Objective value = " + cplex.getObjValue());
					System.out.println("Solution status = " + cplex.getStatus());
					
					System.out.println("Transformed reward function");
					for (int s = 0; s < nStates; s++) {
						for (int a = 0; a < nActions; a++) {
							double r = cplex.getValue(varMap.get(strR2(s, a)));
							System.out.printf("  s%2d, a%2d : %25.20f\n", s, a, r);
						}
					}
					System.out.println();
				}
			}
			
			transformedR = new double[nStates][nActions];
			for (int s = 0; s < nStates; s++) {
				for (int a = 0; a < nActions; a++) {
					transformedR[s][a] = cplex.getValue(varMap.get(strR2(s, a)));
				}
			}

			varMap.clear();
			cplex.clearModel();
		}
		catch (Exception ex) {
			System.err.println(ex);
		}
		
		return transformedR;
	}
	
	private String strPhi(int s) { return "Phi[" + s + "]"; }	
	private String strR2(int s, int a) { return "R2[" + s + "][" + a + "]"; }	
	private String strEps(int s, int a) { return "eps[" + s + "][" + a + "]"; }
	private String strX(int s, int a) { return "X[" + s + "][" + a + "]"; }
	
	public void printReward() {
		for (int a = 0; a < nActions; a++) {
			System.out.printf("Reward: %s\n", pomdp.actions[a]);
			for (int s = 0; s < nStates; s++) {
				System.out.printf("  %s: %.15f\n", pomdp.states[s], transformedR[s][a]);
			}
			System.out.println();
		}		
	}
	
}
