package kr.ac.kaist.pomdp.data;

import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Random;

import kr.ac.kaist.utils.CpuTime;
import kr.ac.kaist.utils.IrlUtil;
import kr.ac.kaist.utils.Mtrx;

import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;

public class FSC {
	public final double INF = 1e5;//Double.POSITIVE_INFINITY;
	
	private ArrayList<FscNode> nodes;
	private ArrayList<Vector>[] nodeBelief;
	private int startNode;
	private double V0;	
	
	private PomdpProblem pomdp;
	private int nStates;
	private int nActions;
	private int nObservs;
	private double gamma;
	private boolean useSparse; 
	
	private IloCplex cplex;
	
	public FSC(PomdpProblem _pomdp) {
		pomdp = _pomdp;
		nStates = _pomdp.nStates;
		nActions = _pomdp.nActions;
		nObservs = _pomdp.nObservations;
		gamma = _pomdp.gamma;
		useSparse = _pomdp.useSparse;
		
		nodes = new ArrayList<FscNode>(); 
		nodeBelief = null;
		
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
			// advanced start switch. default: 1
			//cplex.setParam(IloCplex.IntParam.AdvInd, 2);
		} 
		catch(Exception ex) {}
	}
	
	public FSC(PomdpProblem _pomdp, FSC fsc) {
		pomdp = _pomdp;
		nStates = _pomdp.nStates;
		nActions = _pomdp.nActions;
		nObservs = _pomdp.nObservations;
		gamma = _pomdp.gamma;
		useSparse = _pomdp.useSparse;
		
		nodes = new ArrayList<FscNode>();
		for (int n = 0; n < fsc.size(); n++) 
			nodes.add(fsc.getNode(n).copy());
		
		ArrayList<Vector>[] tmp = fsc.getNodeBelief();
		if (tmp != null) {
			nodeBelief = new ArrayList[nodes.size()];			
			for (int n = 0; n < fsc.size(); n++) {
				nodeBelief[n] = new ArrayList<Vector>();
				for (int b = 0; b < tmp[n].size(); b++) 
					nodeBelief[n].add(((Vector) tmp[n].get(b)).copy());
			}
		}
		
		startNode = fsc.getStartNode();
		V0 = fsc.getV0();		
		
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
			// advanced start switch. default: 1
			//cplex.setParam(IloCplex.IntParam.AdvInd, 2);
		} 
		catch(Exception ex) {}
	}
	
	public FSC copy() {
		FSC fsc = new FSC(pomdp, this);
		return fsc;
	}
	
	public void delete() {
		if (cplex != null) {
			cplex.end();	
			cplex = null;
		}
		if (nodeBelief != null) {
			for (int i = 0; i < nodeBelief.length; i++)
				nodeBelief[i] = null;
			nodeBelief = null;
		}
		if (nodes != null) {
			nodes.clear();
			nodes = null;
		}
	}
	
	public void init(ArrayList<FscNode> _fsc) {
		if (nodes == null) nodes = new ArrayList<FscNode>();
		else nodes.clear();
		Iterator<FscNode> it = _fsc.iterator();
		while (it.hasNext()) {
			FscNode node = it.next();
			nodes.add(node.copy());
		}
		findStartNode();
	}
	
	public int size() 						{ return nodes.size(); }
	
	public void addNode(FscNode node) 		{ nodes.add(node); }
	
	public double getV0() 					{ return V0; }
	
	public int getStartNode() 				{ return startNode; }
	
	public FscNode getNode(int n) 			{ return nodes.get(n); }
	
	public int getAction(int n)				{ return nodes.get(n).act; }
	
	public int getNextNode(int n, int z) 	{ return nodes.get(n).nextNode[z]; }
	
	public ArrayList<Vector>[] getNodeBelief()		{ return nodeBelief; }
	
	// sample the reachable belief states at each node 
	public void sampleNodeBelief(int samplingT, int samplingH, Random rand) {
		int nNodes = nodes.size();
		nodeBelief = new ArrayList[nNodes];
		for (int n = 0; n < nNodes; n++)
			nodeBelief[n] = new ArrayList<Vector>();
		
		for (int i = 0; i < samplingT; i++) {
			Vector b = pomdp.start.copy();
			int s = pomdp.sampleState(b, rand);			
			int n = startNode;
			for (int t = 0; t < samplingH; t++) {
				if (n == FscNode.NO_INFO || n >= nNodes) {
					System.out.println("ERROR : sampleNodeBelief in FSC");
					break;
				}
				if (Mtrx.find(b, nodeBelief[n]) == -1)
					nodeBelief[n].add(b.copy());
					
				int a = nodes.get(n).act;
				int s2 = pomdp.sampleNextState(s, a, rand);
				int z = pomdp.sampleObserv(a, s2, rand);
				b = pomdp.getNextBelief(b, a, z);
				
				n = nodes.get(n).nextNode[z];
				s = s2;
			}
		}
		
		System.out.printf("Reachable beliefs sampled from (%d * %d) trials\n", 
				samplingT, samplingH);
		int totalNum = 0;
		for (int n = 0; n < nNodes; n++) {
			System.out.printf("  node %d : %d\n", n, nodeBelief[n].size());
			totalNum += nodeBelief[n].size();
		}
		System.out.printf("  total # of beliefs : %d\n", totalNum);
		System.out.println();
	}
		
	public void sampleNodeBelief(ArrayList<Vector> bList) {
		int nNodes = nodes.size();
//		nodeBelief = new ArrayList[nNodes];
//		for (int n = 0; n < nNodes; n++)
//			nodeBelief[n] = new ArrayList<Vector>();

		for (Vector b : bList) {		
			int n = FscNode.NO_INFO;
			double maxV = Double.NEGATIVE_INFINITY;
			for (int j = 0; j < nNodes; j++) {
				double v = Mtrx.dot(nodes.get(j).alpha, b);
				if (maxV < v) {
					maxV = v;
					n = j;
				}
			}
			if (nodeBelief[n].size() < 10 && Mtrx.find(b, nodeBelief[n]) == -1) 
				nodeBelief[n].add(b.copy());
		}

		System.out.println("Randomly selected beliefs");
		int totalNum = 0;
		for (int n = 0; n < nNodes; n++) {
			System.out.printf("  node %d : %d\n", n, nodeBelief[n].size());
			totalNum += nodeBelief[n].size();
		}
		System.out.printf("  total # of beliefs : %d\n", totalNum);
		System.out.println();
	}
	
	public void read(String fname) throws Exception {
		BufferedReader br = new BufferedReader(new FileReader(fname));
		ArrayList<String> file = new ArrayList<String>();
		String buf = null;
		String[] bufs = null;
		while ((buf = br.readLine()) != null) {
			int length = buf.length();
			if (length <= 0) continue;
			else file.add(buf);
		}
		br.close();
		
		if (nodes == null) nodes = new ArrayList<FscNode>();
		else nodes.clear();
		Iterator<String> it = file.iterator();
		while (it.hasNext()) {
			buf = it.next();
			bufs = buf.split(" ");
			
			if (bufs.length > 1) {
				int id = Integer.parseInt(bufs[0].trim());
				FscNode node = new FscNode(id, nObservs, nStates, useSparse);
				
				node.act = Integer.parseInt(bufs[1].trim());
				for (int z = 0; z < nObservs; z++)
					node.nextNode[z] = Integer.parseInt(bufs[z + 3].trim());
				nodes.add(node);
			}
		}
		evaluation();
		findStartNode();
	}

	public ArrayList<FscNode> getNodesList() {
		ArrayList<FscNode> newFsc = new ArrayList<FscNode>();
		for (FscNode node : nodes)
			newFsc.add(node.copy());
		return newFsc;
	}
		
	// evaluation the finite state policy using CPLEX
	public void evaluation() {
		try {
			int nNodes = nodes.size();		
			double maxV = Double.NEGATIVE_INFINITY;
			for (int a = 0; a < nActions; a++) {
				Iterator<VectorEntry> itR = Mtrx.Iter(pomdp.R[a]);
				while (itR.hasNext()) {
					VectorEntry veR = itR.next();
					double tmp = Math.abs(veR.get()) / (1.0 - gamma);
					maxV = Math.max(maxV, tmp);
				}
			}
			if (maxV == 0.0) return;
			double minV = -maxV;
			
			int nVars = nNodes * nStates + 1;
			IloNumVar[] vars = new IloNumVar[nVars];
			int vId = 0;
			for (int n = 0; n < nNodes; n++) {
				for (int s = 0; s < nStates; s++) {
					String buf = String.format("v(n%d,s%d)", n, s);
					vars[vId++] = cplex.numVar(minV, maxV, buf);
				}
			}
			vars[vId] = cplex.numVar(0, 0, "eps");
			
			for (int n = 0; n < nNodes; n++) {
				int a = nodes.get(n).act;
				for (int s = 0; s < nStates; s++) {
					IloNumExpr exL = vars[n * nStates + s];				
					IloNumExpr exR = cplex.constant(pomdp.R[a].get(s));				
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
							int n2 = nodes.get(n).nextNode[z];
							if (n2 != FscNode.NO_INFO) {
								IloNumVar V2 = vars[n2 * nStates + s2];
								exR = cplex.sum(exR, cplex.prod(gamma * T * O, V2));
							}
//							else 
//								System.out.println("ERROR : evaluation in FSC!");
						}
					}
					cplex.addEq(exL, exR);
				}
			}
			
			cplex.addMinimize(vars[nNodes * nStates]);
			if (!cplex.solve()) {
				System.out.println("~~~ not optimal in evaluation of FSC.java ~~~");
				System.out.println("Objective value = " + cplex.getObjValue());
				System.out.println("Solution status = " + cplex.getStatus());
				System.out.println("Is primal feasible? " + cplex.isPrimalFeasible());
				System.out.println("Is dual feasible? " + cplex.isDualFeasible());
				
				IloCplex.Quality inf = cplex.getQuality(IloCplex.QualityType.MaxPrimalInfeas);
				double maxinfeas = inf.getValue();
				System.out.printf("Solution quality : %.20f\n", maxinfeas);
			}
			
			for (int n = 0; n < nNodes; n++)
				nodes.get(n).alpha.zero();
			for (int n = 0; n < nNodes; n++) {
				for (int s = 0; s < nStates; s++) {
					double v = cplex.getValue(vars[n * nStates + s]);
					if (v != 0) nodes.get(n).alpha.set(s, v);
				}
			}
			
			vars = null;
			cplex.clearModel();
		}
		catch (Exception ex) {}
	}
	
	// evaluation the finite state policy on reward function R using CPLEX
	public double evaluation(Vector[] R) {
		try {
			int nNodes = nodes.size();
			double maxV = Double.NEGATIVE_INFINITY;
			for (int a = 0; a < nActions; a++) {
				Iterator<VectorEntry> itR = Mtrx.Iter(R[a]);
				while (itR.hasNext()) {
					VectorEntry veR = itR.next();
					double tmp = Math.abs(veR.get()) / (1.0 - gamma);
					maxV = Math.max(maxV, tmp);
				}
			}
			if (maxV == 0) return 0;
			double minV = -maxV;
			
			int nVars = nNodes * nStates + 1;
			IloNumVar[] vars = new IloNumVar[nVars];
			int vId = 0;
			for (int n = 0; n < nNodes; n++) {
				for (int s = 0; s < nStates; s++) {
					String buf = String.format("v(n%d,s%d)", n, s);
					vars[vId++] = cplex.numVar(minV, maxV, buf);
				}
			}
			vars[vId] = cplex.numVar(0, 0, "eps");
			
			for (int n = 0; n < nNodes; n++) {
				int a = nodes.get(n).act;
				for (int s = 0; s < nStates; s++) {
					IloNumExpr exL = vars[n * nStates + s];				
					IloNumExpr exR = cplex.constant(R[a].get(s));			
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
							int n2 = nodes.get(n).nextNode[z];
							if (n2 != FscNode.NO_INFO) {
								IloNumVar V2 = vars[n2 * nStates + s2];
								exR = cplex.sum(exR, cplex.prod(gamma * T * O, V2));
							}
//							else 
//								System.out.println("ERROR : evaluation in FSC!!");
						}
					}
					cplex.addEq(exL, exR);
				}
			}
			
			cplex.addMinimize(vars[nNodes * nStates]);
			if (!cplex.solve()) {
				System.out.println("~~~ not optimal in evaluation with R of FSC.java ~~~");
				System.out.println("Is primal feasible? " + cplex.isPrimalFeasible());
				System.out.println("Is dual feasible? " + cplex.isDualFeasible());
				
				IloCplex.Quality inf = cplex.getQuality(IloCplex.QualityType.MaxPrimalInfeas);
				double maxinfeas = inf.getValue();
				System.out.printf("Solution quality : %.20f\n", maxinfeas);
				System.out.println("Solution status = " + cplex.getStatus());
				System.out.println("Objective value = " + cplex.getObjValue());
			}
			
			for (int n = 0; n < nNodes; n++)
				nodes.get(n).alpha.zero();
			for (int n = 0; n < nNodes; n++) {
				for (int s = 0; s < nStates; s++) {
					double v = cplex.getValue(vars[n * nStates + s]);
					if (v != 0.0) nodes.get(n).alpha.set(s, v);
				}
			}
			
			vars = null;
			cplex.clearModel();
		}
		catch (Exception ex) {
			System.err.println(ex);
		}
		
		ArrayList result = findStartNode();
		int n0 = (Integer) result.get(0);
		double value = (Double) result.get(1);
		return value;
	}
	
	// evaluation the finite state policy using iterative method
	public void evaluation(int maxIters, double minConvError) {
		int nNodes = nodes.size();
		
		Matrix V = Mtrx.Mat(nNodes, nStates, useSparse);
		Matrix V2 = Mtrx.Mat(nNodes, nStates, useSparse);
		Matrix E = Mtrx.Mat(nNodes, nStates, useSparse);		
		V.zero();
		
		double err = Double.POSITIVE_INFINITY;
		for (int t = 0; t < maxIters && err > minConvError; t++) {
			V2.zero();
			E.zero();			
			for (int n = 0; n < nNodes; n++) {
				int a = nodes.get(n).act;
				for (int s = 0; s < nStates; s++) {
					double value = pomdp.R[a].get(s);
					
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
							int n2 = nodes.get(n).nextNode[z];
							if (n2 != FscNode.NO_INFO)
								value += gamma * T * O * V.get(n2, s2);
						}
					}
					if (value != 0.0) V2.set(n, s, value);
				}
			}
			E.set(V2);
			E.add(-1.0, V);
			err = E.norm(Matrix.Norm.Maxvalue);
			V.set(V2);
		}
		
		for (int n = 0; n < nNodes; n++) 
			nodes.get(n).alpha.zero();
		
		Iterator<MatrixEntry> itV2 = Mtrx.Iter(V2);
		while (itV2.hasNext()) {
			MatrixEntry meV2 = itV2.next();
			int n = meV2.row();
			int s = meV2.column();
			nodes.get(n).alpha.set(s, meV2.get());
		}
		
		V = null;
		V2 = null;
		E = null;
	}	
	
	// evaluation the finite state policy on reward function R using iterative method
	public double evaluation(int maxIters, double minConvError, Vector[] R) {
		int nNodes = nodes.size();
		
		Matrix V = Mtrx.Mat(nNodes, nStates, useSparse);
		Matrix V2 = Mtrx.Mat(nNodes, nStates, useSparse);
		Matrix E = Mtrx.Mat(nNodes, nStates, useSparse);		
		V.zero();
		
		double err = Double.POSITIVE_INFINITY;
		for (int t = 0; t < maxIters && err > minConvError; t++) {
			V2.zero();
			E.zero();			
			for (int n = 0; n < nNodes; n++) {
				int a = nodes.get(n).act;
				for (int s = 0; s < nStates; s++) {
					double value = R[a].get(s);
					
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
							int n2 = nodes.get(n).nextNode[z];
							if (n2 != FscNode.NO_INFO)
								value += gamma * T * O * V.get(n2, s2);
						}
					}
					if (value != 0.0) V2.set(n, s, value);
				}
			}
			E.set(V2);
			E.add(-1.0, V);
			err = E.norm(Matrix.Norm.Maxvalue);
			V.set(V2);
		}
		
		for (int n = 0; n < nNodes; n++) 
			nodes.get(n).alpha.zero();
		
		Iterator<MatrixEntry> itV2 = Mtrx.Iter(V2);
		while (itV2.hasNext()) {
			MatrixEntry meV2 = itV2.next();
			int n = meV2.row();
			int s = meV2.column();
			nodes.get(n).alpha.set(s, meV2.get());
		}
		
		V = null;
		V2 = null;
		E = null;
		
		ArrayList result = findStartNode();
		int n0 = (Integer) result.get(0);
		double value = (Double) result.get(1);
		return value;
	}	

	// rearrange the indices of the nodes 
	public void rearrange() {
		int cap = nodes.size() * 2;
		Hashtable<Integer, Integer> label = new Hashtable<Integer, Integer>(cap);
		Iterator<FscNode> it = nodes.iterator();
		for (int i = 0; it.hasNext(); i++) {
			FscNode node = it.next();
			label.put(node.id, i);
		}
		
		ArrayList<FscNode> newFsc = new ArrayList<FscNode>();
		it = nodes.iterator();
		for (int i = 0; it.hasNext(); i++) {
			FscNode newNode = it.next().copy();
			newNode.id = i;
			for (int z = 0; z < nObservs; z++) 
				if (!label.containsKey(newNode.nextNode[z]))
					newNode.nextNode[z] = FscNode.NO_INFO;
				else newNode.nextNode[z] = label.get(newNode.nextNode[z]);
			newFsc.add(newNode);
		}
		init(newFsc);
		label.clear();
		label = null;
		newFsc.clear();
		newFsc = null;
	}
	
	// delete unreachable nodes
	public void delUnreachableNodes(int samplingT, int samplingH, Random rand) {		
		// visiting the reachable nodes
		int nNodes = nodes.size();
		boolean[] bVisit = new boolean[nNodes];
		for (int n = 0; n < nNodes; n++)
			bVisit[n] = false;
		
		bVisit[startNode] = true;
		for (int i = 0; i < samplingT; i++) {
			Vector b = pomdp.start.copy();
			int s = pomdp.sampleState(b, rand);
			int n = startNode;
			for (int t = 0; t < samplingH; t++) {
				if (n == FscNode.NO_INFO || n >= nNodes) break;
				int a = nodes.get(n).act;
				int s2 = pomdp.sampleNextState(s, a, rand);
				int z = pomdp.sampleObserv(a, s2, rand);
				b = pomdp.getNextBelief(b, a, z);				
				n = nodes.get(n).nextNode[z];
				bVisit[n] = true;
				s = s2;
			}
		}
		
		ArrayList<FscNode> newFsc = new ArrayList<FscNode>();
		for (int i = 0; i < nodes.size(); i++) 
			if (bVisit[i]) newFsc.add(nodes.get(i).copy());
		init(newFsc);
		bVisit = null;
		newFsc.clear();
		newFsc = null;
	}	
	
	// delete unreachable nodes
	public void delUnreachableNodes() {		
		// visiting the reachable nodes
		int nNodes = nodes.size();
		boolean[] bVisit = new boolean[nNodes];
		bVisit[startNode] = true;
		ArrayList<Integer> curNode = new ArrayList<Integer>();
		curNode.add(startNode);
		while(curNode.size() > 0) {
			int n = curNode.get(0);
			curNode.remove(0);
			if (n != FscNode.NO_INFO) {
				for (int z = 0; z < nObservs; z++) {
					int n2 = nodes.get(n).nextNode[z];
					if (n2 != FscNode.NO_INFO && !bVisit[n2]) {
						bVisit[n2] = true;
						curNode.add(n2);
					}
				}	
			}
		}
		
		ArrayList<FscNode> newFsc = new ArrayList<FscNode>();
		for (int i = 0; i < nodes.size(); i++) 
			if (bVisit[i]) newFsc.add(nodes.get(i).copy());
		init(newFsc);
		bVisit = null;
		newFsc.clear();
		newFsc = null;
	}

	public ArrayList<Vector>[] findReachableBeliefs() {		
		// visiting the reachable nodes
		int nNodes = nodes.size();
		ArrayList<Integer> curNode = new ArrayList<Integer>();
		ArrayList<Vector> curB = new ArrayList<Vector>();
		ArrayList<Vector>[] nodeBeliefs = new ArrayList[nNodes];
		for (int n = 0; n < nNodes; n++)
			nodeBeliefs[n] = new ArrayList<Vector>();
		
		curNode.add(startNode);
		curB.add(pomdp.start.copy());
		while(curNode.size() > 0) {			
			int n = curNode.get(0);
			curNode.remove(0);
			Vector b = curB.get(0);
			curB.remove(0);
			
			if (Mtrx.find(b, nodeBeliefs[n]) != -1) continue;
			nodeBeliefs[n].add(b);
			int a = getAction(n);
			for (int z = 0; z < nObservs; z++) {
				boolean bReachable = false;
				for (int s = 0; s < nStates; s++) {
					if (b.get(s) > 0) {
						for (int s2 = 0; s2 < nStates; s2++) {
							double T = pomdp.T[a][s].get(s2);
							double O = pomdp.O[a][s2].get(z);
							if (T * O > 0) {
								bReachable = true;
								break;
							}
						}
					}
					if (bReachable) break;
				}
				if (bReachable) {
					int n2 = nodes.get(n).nextNode[z];
					Vector b2 = pomdp.getNextBelief(b, a, z);
					if (n2 != FscNode.NO_INFO) {
						curNode.add(n2);
						curB.add(b2);
					}
				}
			}
		}

//		System.out.println("List of reachable beliefs");
//		for (int n = 0; n < nNodes; n++) {
//			System.out.printf("  node %d : %d\n", n, nodeBeliefs[n].size());
//			for (Vector b : nodeBeliefs[n]) {
//				System.out.print("  [");
//				for (int s = 0; s < nStates; s++) System.out.printf("%f ", b.get(s));
//				System.out.println("]");
//			}
//		}
//		System.out.println();
//		
//		ArrayList<Vector> bSet = new ArrayList<Vector>();
//		for (int n = 0; n < nNodes; n++)
//			for (Vector b : nodeBeliefs[n])
//				if (Mtrx.find(b, bSet) == -1) bSet.add(b);		
//		Vector[] bList = new Vector[bSet.size()];
//		for (int i = 0; i < bSet.size(); i++) bList[i] = bSet.get(i);
//		System.out.printf("# of reachable beliefs : %d\n", bSet.size());		
		
//		return bList;
		return nodeBeliefs;
	}
	
	// return id of start node and its value
	public ArrayList findStartNode() {
		int startN = FscNode.NO_INFO;
		double maxValue = Double.NEGATIVE_INFINITY;
		Vector b0 = pomdp.start.copy();
		for (FscNode node : nodes) {
			double v = Mtrx.dot(b0, node.alpha);
			if (v > maxValue) {
				maxValue = v;
				startN = node.id;
			}
		}
		b0 = null;
		startNode = startN;
		V0 = maxValue;
		ArrayList result = new ArrayList(2);
		result.add(startN);
		result.add(maxValue);
		return result;
	}
	
	// generate new nodes by dynamic programming backup
	public ArrayList<FscNode> dpBackup() {
		int nNodes = nodes.size();		
		ArrayList<FscNode> dpList = new ArrayList<FscNode>();
		int[][] etaList = IrlUtil.combination(nNodes, nObservs);
		//  Dp-Backup
		for (int a = 0; a < nActions; a++) {
			for (int n = 0; n < etaList.length; n++) {
				FscNode node = new FscNode(nObservs, nStates, useSparse);
				node.act = a;
				for (int z = 0; z < nObservs; z++)
					node.nextNode[z] = etaList[n][z];
				if (Mtrx.find(node, dpList) == FscNode.NO_INFO)
					dpList.add(node);
			}
		}
		etaList = null;
		return dpList;
	}
	
	// generate new nodes by witness thm.
	public ArrayList<FscNode> wBackup() {
		int nNodes = nodes.size();		
		boolean[] bActions = new boolean[nActions];
		for (int a = 0; a < nActions; a++) bActions[a] = false;

		// Dp-Backup using Witness Thm.
		ArrayList<FscNode> dpList = new ArrayList<FscNode>();
		for (int n = 0; n < nNodes; n++) {
			int a = nodes.get(n).act;
			bActions[a] = true;
			FscNode bestNode = nodes.get(n).copy();
			for (int z = 0; z < nObservs; z++) {
				for (int n2 = 0; n2 < nNodes; n2++) {
					FscNode tmpNode = bestNode.copy();
					tmpNode.nextNode[z] = n2;
					if (Mtrx.find(tmpNode, dpList) == FscNode.NO_INFO)
						dpList.add(tmpNode);
				}
			}
		}
		
		boolean bFullDp = false;
		for (int a = 0; a < nActions; a++) { 
			if (!bActions[a]) {
				bFullDp = true;
				break;
			}
		}
		if (bFullDp) {
			int[][] etaList = IrlUtil.combination(nNodes, nObservs);
			//  Dp-Backup
			for (int a = 0; a < nActions; a++) {
				if (!bActions[a]) {
					for (int n = 0; n < etaList.length; n++) {
						FscNode tmpNode = new FscNode(nObservs, nStates, useSparse);
						tmpNode.act = a;
						for (int z = 0; z < nObservs; z++)
							tmpNode.nextNode[z] = etaList[n][z];
						if (Mtrx.find(tmpNode, dpList) == FscNode.NO_INFO)
							dpList.add(tmpNode);
					}
				}
			}
			etaList = null;
		}
		bActions = null;
		return dpList;
	}
	
	// calculate occupancy distribution using CPLEX
	public Matrix calOccDist() throws Exception {
		int nNodes = size();		
						
		IloNumVar[] vars = new IloNumVar[nNodes * nStates + 1];
		for (int s = 0; s < nStates; s++) {
			for (int n = 0; n < nNodes; n++) {
				String name = String.format("occ[s%d][n%d]", s, n);
				vars[s * nNodes + n] = cplex.numVar(0, INF, name);
			}
		}
		int nEps = nNodes * nStates;
		vars[nEps] = cplex.numVar(0, 0, "eps");
		
		for (int s2 = 0; s2 < nStates; s2++) {
			for (int n2 = 0; n2 < nNodes; n2++) {
				IloNumExpr expOcc = cplex.numExpr();
				if (n2 == getStartNode()) 
					expOcc = cplex.sum(expOcc, pomdp.start.get(s2));
				
				for (int s = 0; s < nStates; s++) {
					for (int n = 0; n < nNodes; n++) {
						int a = getAction(n);
						double T = pomdp.T[a][s].get(s2);
						Iterator<VectorEntry> itO = Mtrx.Iter(pomdp.O[a][s2]);
						while (itO.hasNext()) {
							VectorEntry veO = itO.next();
							int z = veO.index();
							double O = veO.get();
							if (n2 == getNextNode(n, z))
								expOcc = cplex.sum(expOcc, 
										cplex.prod(gamma * T * O, vars[s * nNodes + n]));
						}
					}
				}
				cplex.addEq(vars[s2 * nNodes + n2], expOcc);
				/*expOcc = cplex.sum(vars[s2 * nNodes + n2], cplex.prod(-1.0, expOcc));
				cplex.addLe(expOcc, vars[nEps]);
				cplex.addLe(cplex.prod(-1.0, vars[nEps]), expOcc);*/
			}
		}
		cplex.addMinimize(vars[nEps]);
		cplex.solve();
		
		Matrix occ = Mtrx.Mat(nStates, nNodes, useSparse);
		for (int s = 0; s < nStates; s++) {
			for (int n = 0; n < nNodes; n++) {
				double x = cplex.getValue(vars[s * nNodes + n]);
				if (x != 0) occ.set(s, n, x);
			}
		}

		vars = null;
		cplex.clearModel();
		return occ;
	}
	
	// calculate occupancy distribution using iterative method
	public Matrix calOccDist2(FSC fsc) {
		int iter = 1000;
		double minConvError = 1e-6;
		int nNodes = fsc.size();		
		
		Matrix oldOcc = Mtrx.Mat(nStates, nNodes, true);
		Matrix newOcc = Mtrx.Mat(nStates, nNodes, true);
		Matrix E = Mtrx.Mat(nStates, nNodes, true);
		double err = Double.POSITIVE_INFINITY;
		for (int t = 0; t < iter && err > minConvError; t++) {
			newOcc.zero();
			E.zero();
			for (int s2 = 0; s2 < nStates; s2++) {
				for (int n2 = 0; n2 < nNodes; n2++) {
					double sum = 0.0;
					Iterator<MatrixEntry> itM = Mtrx.Iter(oldOcc);
					while (itM.hasNext()) {
						MatrixEntry meM = itM.next();
						int s = meM.row();
						int n = meM.column();
						double occ = meM.get();
						int a = fsc.getAction(n);
						double T = pomdp.T[a][s].get(s2);
						Iterator<VectorEntry> itO = Mtrx.Iter(pomdp.O[a][s2]);
						while (itO.hasNext()) {
							VectorEntry veO = itO.next();
							int z = veO.index();
							double O = veO.get();
							if (n2 == fsc.getNextNode(n, z))
								sum += gamma * occ * T * O;
						}
					}
					if (n2 == fsc.getStartNode())
						sum += pomdp.start.get(s2);
					if (sum != 0.0)
						newOcc.set(s2, n2, sum);
				}
			}
			E.set(newOcc);
			E.add(-1.0, oldOcc);
			err = E.norm(Matrix.Norm.Maxvalue);
			oldOcc.set(newOcc);
		}
		newOcc = null;
		E = null;
		return oldOcc;
	}
	
	public void simulate(int maxIters, int maxSteps, boolean bPrint, Random rand) {
		double startTime = CpuTime.getCurTime();
		System.out.println("== Simulation ======================================================");
		double expValue = 0.0;
		
		for (int i = 0; i < maxIters; i++) {
			Vector b = pomdp.start.copy();
			int s = pomdp.sampleState(b, rand);
			int n = startNode;
			double v = 0.0;
			if (bPrint) System.out.printf("%d - th trajectory\n", i);
			for (int t = 0; t < maxSteps; t++) {
				if (n == FscNode.NO_INFO) break;
				int a = nodes.get(n).act;
				int s2 = pomdp.sampleNextState(s, a, rand);
				int z = pomdp.sampleObserv(a, s2, rand);
				v += Math.pow(pomdp.gamma, t) * pomdp.R[a].get(s);
				
				if (bPrint) 
//					System.out.printf("  [%d]%s ->\t%s ->\t%s / %s\t: %f\n", 
//						n, pomdp.states[s], pomdp.actions[a], pomdp.states[s2], pomdp.observations[z], pomdp.R[a].get(s));
					System.out.printf("  [%d]s%d ->\t%s ->\ts%d / %s\t: %f\n", 
						n, s, pomdp.actions[a], s2, pomdp.observations[z], pomdp.R[a].get(s));
				
				b = pomdp.getNextBelief(b, a, z);
				s = s2;
				n = nodes.get(n).nextNode[z];
				//if (bPrint) System.out.printf("a%d z%d -> s%d n%d", a, z, s, n);
			}
			if (bPrint) System.out.printf(" :: %f\n", v);
			expValue += v;
		}
		expValue /= maxIters;
		
		double elapsedTime = CpuTime.getElapsedTime(startTime);
		System.out.printf("Elapsed Time     : %.4f sec\n", elapsedTime);
		System.out.printf("Expected Value   : %f (%f)\n", expValue, V0);
	}
	
	public void simulate(Vector[] R, int maxIters, int maxSteps, boolean bPrint, Random rand) {
		double startTime = CpuTime.getCurTime();
		System.out.println("== Simulation ======================================================");
		double expValue = 0.0;
		
		for (int i = 0; i < maxIters; i++) {
			Vector b = pomdp.start.copy();
			int s = pomdp.sampleState(b, rand);
			int n = startNode;
			double v = 0.0;
			if (bPrint) System.out.printf("%d - th trajectory\n", i);
			for (int t = 0; t < maxSteps; t++) {
				if (n == FscNode.NO_INFO) break;
				int a = nodes.get(n).act;
				int s2 = pomdp.sampleNextState(s, a, rand);
				int z = pomdp.sampleObserv(a, s, rand);
				v += Math.pow(pomdp.gamma, t) * R[a].get(s);
				
				if (bPrint) 
					System.out.printf("  %s -> %s -> %s, %s : %f\n", 
						pomdp.states[s], pomdp.actions[a], pomdp.states[s2], pomdp.observations[z], pomdp.R[a].get(s));
				
				b = pomdp.getNextBelief(b, a, z);
				s = s2;
				n = nodes.get(n).nextNode[z];
				//if (bPrint) System.out.printf("a%d z%d -> s%d n%d", a, z, s, n);
			}
			if (bPrint) System.out.printf(" :: %f\n", v);
			expValue += v;
		}
		expValue /= maxIters;
		
		double elapsedTime = CpuTime.getElapsedTime(startTime);
		System.out.printf("Elapsed Time     : %.4f sec\n", elapsedTime);
		System.out.printf("Expected Value   : %f (%f)\n", expValue, V0);
	}
	
	public double calMaxV(Vector b) {
		double maxV = Double.NEGATIVE_INFINITY;
		for (FscNode node : nodes)
			maxV = Math.max(maxV, Mtrx.dot(b, node.alpha));
		return maxV;
	}
	
	public double calV(int n, Vector b) {
		return Mtrx.dot(b, nodes.get(n).alpha);
	}
			
	public void print() {
		if (nodes.size() > 100) 
			System.out.printf("# of nodes in FSC : %d\n\n", nodes.size());
		else {
			System.out.println("== Policy ==========================================================");
			System.out.printf("Start Node: n%d\n", startNode);
			for (FscNode node : nodes) {
				if (pomdp.actions == null) System.out.printf("n%d: %d\n", node.id, node.act);
				else System.out.printf("n%d: %s\n", node.id, pomdp.actions[node.act]);
				System.out.print("    ");
				for (int z = 0; z < pomdp.nObservations; z++)
					if (node.nextNode[z] == FscNode.NO_INFO) {
						if (pomdp.observations == null) System.out.printf("%d->n* ", z);
						else System.out.printf("%s->n* ", pomdp.observations[z]);
					}
					else {
						if (pomdp.observations == null) System.out.printf("%d->n%d ", z, node.nextNode[z]);
						else System.out.printf("%s->n%d ", pomdp.observations[z], node.nextNode[z]);
					}
				System.out.println();
				//			System.out.print("    ");
				//			for (int s = 0; s < pomdp.nStates; s++)
				//				System.out.printf("%f ", node.alpha.get(s));
				//			System.out.println();
			}			
			System.out.println();
		}
	}

	public void print(String filename) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
		bw.write(String.format("%d\n", nodes.size()));
		for (FscNode node : nodes) {
			bw.write(String.format("%d %d  ", node.id, node.act));
			for (int z = 0; z < nObservs; z++)
				bw.write(String.format("%d ", node.nextNode[z]));
			bw.write(String.format("\n"));
		}
		bw.close();
	}
	
//	public void sampleNodeBelief(int nBeliefs, Random rand) {
//		int nNodes = nodes.size();
//		nodeBelief = new ArrayList[nNodes];
//		for (int n = 0; n < nNodes; n++) 
//			nodeBelief[n] = new ArrayList<Vector>();
//		
//		for (int t = 0; t < nNodes * nBeliefs * 100; t++) {
//			Vector b = Mtrx.Vec(nStates, true);
//			for (int i = 0; i < nStates; i++) 
//				b.set(i, rand.nextDouble());
//			b = Mtrx.normalize(b);
//			
//			int n = FscNode.NO_INFO;
//			double maxV = Double.NEGATIVE_INFINITY;
//			for (int j = 0; j < nNodes; j++) {
//				double v = Mtrx.dot(nodes.get(j).alpha, b);
//				if (maxV < v) {
//					maxV = v;
//					n = j;
//				}
//			}
//			if (nodeBelief[n].size() < nBeliefs && Mtrx.find(b, nodeBelief[n]) == -1) 
//				nodeBelief[n].add(b.copy());
//			
//			boolean bExit = true;
//			for (int j = 0; j < nNodes; j++) {
//				if (nodeBelief[j].size() < nBeliefs) {
//					bExit = false;
//					break;
//				}
//			}
//			if (bExit) break;
//		}
//		
//		int totalNum = 0;
//		for (int n = 0; n < nNodes; n++) {
//			System.out.printf("node %d : %d\n", n, nodeBelief[n].size());
//			totalNum += nodeBelief[n].size();
//		}
//		System.out.printf("# of sampled beliefs : %d\n", totalNum);
//		System.out.println();
//	}
	
//	public void sampleNodeBelief(ArrayList<Vector> bList) {
//		int nNodes = nodes.size();
//		nodeBelief = new ArrayList[nNodes];
//		for (int n = 0; n < nNodes; n++)
//			nodeBelief[n] = new ArrayList<Vector>();
//		
//		for (Vector b : bList) {		
//			int n = FscNode.NO_INFO;
//			double maxV = Double.NEGATIVE_INFINITY;
//			for (int j = 0; j < nNodes; j++) {
//				double v = Mtrx.dot(nodes.get(j).alpha, b);
//				if (maxV < v) {
//					maxV = v;
//					n = j;
//				}
//			}
//			if (Mtrx.find(b, nodeBelief[n]) == -1) 
//				nodeBelief[n].add(b.copy());
//		}
//		
//		System.out.println("Randomly selected beliefs");
//		int totalNum = 0;
//		for (int n = 0; n < nNodes; n++) {
//			System.out.printf("node %d : %d\n", n, nodeBelief[n].size());
//			totalNum += nodeBelief[n].size();
//		}
//		//System.out.printf("# of sampled beliefs : %d\n", totalNum);
//		System.out.println();
//	}

	// sample the reachable belief states at each node 
//	public void sampleNodeBelief(int samplingT, int samplingH, int minNBeliefs, Random rand) {
//		int nNodes = nodes.size();
//		nodeBelief = new ArrayList[nNodes];
//		for (int n = 0; n < nNodes; n++)
//			nodeBelief[n] = new ArrayList<Vector>();
//		
//		for (int i = 0; i < samplingT; i++) {
//			Vector b = pomdp.start.copy();
//			int s = pomdp.sampleState(b, rand);
//			int n = startNode;
//			for (int t = 0; t < samplingH; t++) {
//				if (n == FscNode.NO_INFO || n >= nNodes) break;
//				if (Mtrx.find(b, nodeBelief[n]) == -1) { 
//					nodeBelief[n].add(b.copy());
////					for (int a = 0; a < nActions; a++) {
////						for (int z = 0; z < nObservs; z++) {
////							Vector b2 = pomdp.getNextBelief(b, a, z);
////							int n2 = FscNode.NO_INFO;
////							double maxV = Double.NEGATIVE_INFINITY;
////							for (int j = 0; j < nNodes; j++) {
////								double v = Mtrx.dot(nodes.get(j).alpha, b2);
////								if (maxV < v) {
////									maxV = v;
////									n2 = j;
////								}
////							}
////							if (Mtrx.find(b2, nodeBelief[n2]) == -1) 
////								nodeBelief[n2].add(b2.copy());
////						}
////					}
//				}
//				int a = nodes.get(n).act;
//				int s2 = pomdp.sampleNextState(s, a, rand);
//				int z = pomdp.sampleObserv(a, s2, rand);
//				b = pomdp.getNextBelief(b, a, z);
//				
//				n = nodes.get(n).nextNode[z];
//				s = s2;
//			}
//		}
//		
//		System.out.println("Reachable beliefs");
//		int totalNum = 0;
//		for (int n = 0; n < nNodes; n++) {
//			System.out.printf("node %d : %d\n", n, nodeBelief[n].size());
//			totalNum += nodeBelief[n].size();
//		}
//		//System.out.printf("# of sampled beliefs : %d\n", totalNum);
//		System.out.println();
//		
//		for (int t = 0; t < nNodes * minNBeliefs * 1000; t++) {
//			Vector b = Mtrx.Vec(nStates, true);
//			for (int i = 0; i < nStates; i++) 
//				b.set(i, rand.nextDouble());
//			b = Mtrx.normalize(b);
//			
//			int n = FscNode.NO_INFO;
//			double maxV = Double.NEGATIVE_INFINITY;
//			for (int j = 0; j < nNodes; j++) {
//				double v = Mtrx.dot(nodes.get(j).alpha, b);
//				if (maxV < v) {
//					maxV = v;
//					n = j;
//				}
//			}
//			if (nodeBelief[n].size() < minNBeliefs && Mtrx.find(b, nodeBelief[n]) == -1) 
//				nodeBelief[n].add(b.copy());
//			
//			boolean bExit = true;
//			for (int j = 0; j < nNodes; j++) {
//				if (nodeBelief[j].size() < minNBeliefs) {
//					bExit = false;
//					break;
//				}
//			}
//			if (bExit) break;
//		}
//		
//		System.out.println("Randomly selected beliefs");
//		totalNum = 0;
//		for (int n = 0; n < nNodes; n++) {
//			System.out.printf("node %d : %d\n", n, nodeBelief[n].size());
//			totalNum += nodeBelief[n].size();
//		}
//		//System.out.printf("# of sampled beliefs : %d\n", totalNum);
//		System.out.println();
//	}
}

