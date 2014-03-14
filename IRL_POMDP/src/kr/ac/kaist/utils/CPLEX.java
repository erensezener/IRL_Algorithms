package kr.ac.kaist.utils;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Hashtable;

import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

/**
 * Class for wrapping CPLEX
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class CPLEX {
	private IloCplex cplex;
	private IloNumVar[] vars;
	private Hashtable varNameMap;
	
	public CPLEX(int nVars) throws Exception {
		vars = new IloNumVar[nVars];
		varNameMap = new Hashtable();
		cplex = new IloCplex();
		
		// Set simplex iteration information display
		// 0 : No iteration messages until solution
		// 1 : Iteration information after each refactoring; default
		// 2 : Iteration information for each iteration
		cplex.setParam(IloCplex.IntParam.SimDisplay, 0);
		
		// Turn off display of messages to screen
		cplex.setOut(null);
		
		// Sets the maximum number of simplex iterations
		// Any nonnegative integer; default: 2100000000.
		//cplex.setParam(IloCplex.IntParam.ItLim, 2100000000);
	}
	
	public void delete() throws Exception { 
		cplex.clearModel(); 
		cplex.end();
	}
	
	public double[] solve(boolean bPrint) throws Exception {
		double[] solution = new double[vars.length];
		if (cplex.solve()) {
			solution = cplex.getValues(vars);
			if (bPrint) {
				System.out.println("-- Result of CPLEX --");
				System.out.println("Objective value = " + cplex.getObjValue());
				System.out.println("Solution status = " + cplex.getStatus());
				for (int i = 0; i < vars.length; i++) 
					System.out.println(vars[i].getName() + " = " + cplex.getValue(vars[i]));
			}
		}
		else {
			System.out.println("Solution status = " + cplex.getStatus());
		}
		return solution;
	}
	
	public double getObjVal() throws Exception {
		double objVal = cplex.getObjValue();
		return objVal;
	}
	
	public double[] getSolution() throws Exception {
		double[] solution = new double[vars.length];
		solution = cplex.getValues(vars);
		return solution;
	}
	
	public double getSolution(String var) throws Exception {
		double sol = cplex.getValue((IloNumVar) varNameMap.get(var));
		return sol;
	}
	
	public void setVariable(int id, double lb, double ub, String name)
	throws Exception {
		vars[id] = cplex.numVar(lb, ub, name);
		varNameMap.put(name, vars[id]);
	}
	
	public void setConst(double[] coefs, int[] ids, String ieq, double rhs) 
	throws Exception {
		IloNumExpr expr = mkExpr(coefs, ids);
		
		if (ieq.equals("GE")) cplex.addGe(expr, rhs);
		else if (ieq.equals("LE")) cplex.addLe(expr, rhs);
		else if (ieq.equals("EQ")) cplex.addEq(expr, rhs);
		else throw new Exception("The sign of inequaility is not properly specified.");
	}
	
	public void setConst(ArrayList<Double> coefList, ArrayList<String> varList, String ieq, double rhs) 
	throws Exception {
		IloNumExpr expr = mkExpr(coefList, varList);
		
		if (ieq.equals("GE")) cplex.addGe(expr, rhs);
		else if (ieq.equals("LE")) cplex.addLe(expr, rhs);
		else if (ieq.equals("EQ")) cplex.addEq(expr, rhs);
		else throw new Exception("The sign of inequaility is not properly specified.");
	}
	
	public void setObjFun(double[] coefs, int[] ids, String str)
	throws Exception {
		IloNumExpr expr = mkExpr(coefs, ids);
		
		if (str.equals("MAX")) cplex.addMaximize(expr);
		else if (str.equals("MIN")) cplex.addMinimize(expr);
		else throw new Exception("Objective function is not properly defined.");
	}
	
	public void setObjFun(ArrayList<Double> coefList, ArrayList<String> varList, String str)
	throws Exception {
		IloNumExpr expr = mkExpr(coefList, varList);
		
		if (str.equals("MAX")) cplex.addMaximize(expr);
		else if (str.equals("MIN")) cplex.addMinimize(expr);
		else throw new Exception("Objective function is not properly defined.");
	}
	
	// min_x 1/2 x^T H x + f^T x such that A x <= b, Aeq x = beq, lb <= x <= ub
	public void setObjFun(double[][] H, double[] f, double[] A, double b, 
			double[] Aeq, double beq, double[] lb, double[] ub) throws Exception {
		int n = lb.length;
		vars = cplex.numVarArray(n, lb, ub);
		cplex.addLe(cplex.scalProd(A, vars), b);
		cplex.addEq(cplex.scalProd(Aeq, vars), beq);
		
		IloNumExpr expr = cplex.scalProd(f, vars);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				cplex.sum(expr, cplex.prod(H[i][j] / 2.0, vars[i], vars[j]));
			}
		}
		cplex.addMinimize(expr);
		
		// Markowitz tolerance (0.0001 ~ 0.99999, default : 0.01)
		// increasing the threshold may improve the numerical properties of the solutions
		cplex.setParam(IloCplex.DoubleParam.EpMrk, 0.3);
	}	
	
	// min_x 1/2 x^T H x + f^T x such that Aeq x = beq, lb <= x <= ub
	public void setObjFun(double[][] H, double[] f, double[] Aeq, double beq, double[] lb, double[] ub) 
	throws Exception {
		int n = lb.length;
		vars = cplex.numVarArray(n, lb, ub);		
		cplex.addEq(cplex.scalProd(Aeq, vars), beq);		
		IloNumExpr expr = cplex.scalProd(f, vars);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				expr = cplex.sum(expr, cplex.prod(H[i][j] / 2.0, vars[i], vars[j]));
			}
		}
		cplex.addMinimize(expr);
	}	
	
	public IloNumExpr mkExpr(double[] coefs, int[] ids) throws Exception {
		if (coefs.length != ids.length)
			throw new Exception ("Variables and coefficients are not properly matched.");
		IloNumVar[] tmpVars = new IloNumVar[ids.length];
		for (int i = 0; i < ids.length; i++)
			tmpVars[i] = vars[ids[i]];
		IloNumExpr expr = cplex.scalProd(coefs, tmpVars);
		return expr;
	}
	
	public IloNumExpr mkExpr(ArrayList<Double> coefList, ArrayList<String> varList)  
	throws Exception {
		if (coefList.size() != varList.size())
			throw new Exception ("Variables and coefficients are not properly matched.");
		double[] tmpCoefs = new double[coefList.size()];
		IloNumVar[] tmpVars = new IloNumVar[varList.size()];
		for (int i = 0; i < varList.size(); i++) {
			tmpCoefs[i] = coefList.get(i);
			tmpVars[i] = (IloNumVar) varNameMap.get(varList.get(i));
		}
		IloNumExpr expr = cplex.scalProd(tmpCoefs, tmpVars);
		return expr;
	}
	
	public int getNrows() { return cplex.getNrows(); }
	public int getNcols() { return cplex.getNcols(); }
	
	public int findVarId(String var) {
		return (Integer) varNameMap.get(var);
	}
	
	public void printVariables() throws Exception {
		for (int i = 0; i < vars.length; i++) {
			System.out.println(vars[i].getName() 
					+ " : Lower bound = " + vars[i].getLB()
					+ " : Upper bound = " + vars[i].getUB());
		}
	}
	
	public void fprintModel(String fname) throws Exception {
		cplex.exportModel(fname);
	}

	public void fprintSol(String fname) throws Exception {
		cplex.writeSolution(fname);
	}
	
	public void fprintSolution(String fname) throws Exception {		
		BufferedWriter bw = new BufferedWriter(new FileWriter(fname));
		bw.write("-- Result of CPLEX --");
		bw.write("Objective value = " + cplex.getObjValue());
		bw.write("Solution status = " + cplex.getStatus());
		for (int i = 0; i < vars.length; i++) 
			bw.write(vars[i].getName() + " = " + cplex.getValue(vars[i]));		
		bw.close();
	}

}