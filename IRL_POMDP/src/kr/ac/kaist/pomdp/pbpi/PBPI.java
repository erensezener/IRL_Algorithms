package kr.ac.kaist.pomdp.pbpi;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;

import kr.ac.kaist.pomdp.data.FSC;
import kr.ac.kaist.pomdp.data.FscNode;
import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.utils.CpuTime;
import kr.ac.kaist.utils.Mtrx;

/**
 * Point-based policy iteration (PBPI)
 * 
 * @see S. Ji, R. Parr, H. Li et al., Point-based policy iteration, AAAA 2007.
 *      
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class PBPI {	
	private int maxIters = 1000;
	private double minConvError = 1e-10;
	private boolean bPrint = false;
	
	private PomdpProblem pomdp;
	private int nStates;
	private int nActions;
	private int nObservs;
	private double gamma;
	private boolean useSparse;
	
	private ArrayList<Vector> beliefSet;
	private FSC fsc;
	private int startNode;
	private double V0;	
	private double totalElapsedTime;
	
	public PBPI(PomdpProblem _pomdp, ArrayList<Vector> _beliefSet) {
		pomdp = _pomdp;
		nStates = _pomdp.nStates;
		nActions = _pomdp.nActions;
		nObservs = _pomdp.nObservations;
		gamma = _pomdp.gamma;
		useSparse = _pomdp.useSparse;
		beliefSet = _beliefSet;
	}
	
	public void delete() {
		fsc.delete();
	}
	
	public void setParams(int _maxIters, double _minConvError) {
		maxIters = _maxIters;
		minConvError = _minConvError;
	}
	
	// make the initial policy
	private FSC initFsc() {
		Vector b = pomdp.start.copy();
		double maxR = Double.NEGATIVE_INFINITY;
		int maxA = -1;
		for (int a = 0; a < nActions; a++) {
			double r = Mtrx.dot(b, pomdp.R[a]);
			if (r > maxR) {
				maxR = r;
				maxA = a;
			}			
		}
		FscNode node = new FscNode(0, nObservs, nStates, useSparse);
		node.act = maxA;
		node.alpha = pomdp.R[maxA].copy();
		
		FSC fsc = new FSC(pomdp);
		fsc.addNode(node);
		return fsc;
	}

	public FSC run(boolean _bPrint, Random rand) {
		double T0 = CpuTime.getCurTime();
		bPrint = _bPrint;
		if (bPrint) System.out.println("== Run PBPI ========================================================");
		
		double initV = Double.NEGATIVE_INFINITY;
		double oldV = Double.NEGATIVE_INFINITY;
		double newV = Double.NEGATIVE_INFINITY;
		fsc = initFsc();
		
		for (int t = 0; t < maxIters; t++) {
			double T1 = CpuTime.getCurTime();
			if (bPrint) System.out.printf("### %d-Iteration\n", t);
			
			double T2 = CpuTime.getCurTime();
			fsc.evaluation(); //fsc.evaluation(maxIters, minConvError);
			if (bPrint)	{
				System.out.printf("  Fsc Evaluation : %d nodes in FSC\n", fsc.size());
				System.out.printf("    Elapsed Time: %.2f sec\n", CpuTime.getElapsedTime(T2));
			}
			
			ArrayList<FscNode> backup = improvement(fsc);
			ArrayList<FscNode> newPolicy = fscTransformation(fsc, backup);		
			fsc = pruneFsc(newPolicy, backup);
			release(backup);
			release(newPolicy);
			
			T2 = CpuTime.getCurTime();
			if (bPrint)	System.out.println("  Fsc Rearrange");
			fsc.rearrange();
			if (bPrint)	System.out.printf("    Elapsed Time: %.2f sec\n", CpuTime.getElapsedTime(T2));
						
			// check the convergence condition
			newV = calExpV(fsc);
			if (bPrint) System.out.printf("  Size:%d,  V:%f (%f),  Time:%.2f sec\n\n", 
					fsc.size(), newV, fsc.getV0(), CpuTime.getElapsedTime(T1));
						
			if (t == 0) initV = newV;
			else if (Math.abs(newV - oldV) < minConvError * Math.abs(initV - newV)) break;
			oldV = newV;
		}		
		double T1 = CpuTime.getCurTime();
		fsc.evaluation();
		if (bPrint)	{
			System.out.printf("  Fsc Evaluation : %d nodes in FSC\n", fsc.size());
			System.out.printf("    Elapsed Time: %.2f sec\n", CpuTime.getElapsedTime(T1));
		}
		
		T1 = CpuTime.getCurTime();
		if (bPrint)	System.out.println("  Final Fsc Rearrange");
		fsc.findStartNode();
		fsc.delUnreachableNodes(fsc.size() * 10, 100, rand); //fsc.delUnreachableNodes();
		fsc.rearrange();
				
		fsc.evaluation();
		fsc.findStartNode();
		startNode = fsc.getStartNode();
		V0 = fsc.getV0();
		if (bPrint)	System.out.printf("    Elapsed Time: %.2f sec\n", CpuTime.getElapsedTime(T1));
		
		totalElapsedTime = CpuTime.getElapsedTime(T0);
		if (bPrint) {
			System.out.println("====================================================================");
			System.out.printf("Elapsed Time: %.2f sec\n", totalElapsedTime);
			System.out.printf("Expected Value   : %f\n\n", fsc.calMaxV(pomdp.start));
		}
		return fsc;
	}
	
	private ArrayList<FscNode> improvement(FSC fsc) {
		double startTime = CpuTime.getCurTime();
		if (bPrint)	System.out.printf("  Improvement : %d nodes in FSC\n", fsc.size());
		
		int newId = fsc.size();
		ArrayList<FscNode> backup = new ArrayList<FscNode>();
		for (Vector b : beliefSet) {
			FscNode newNode = pbSingleBackup(fsc, b);
			if (Mtrx.find(newNode, backup) == FscNode.NO_INFO) {
				newNode.id = newId++;
				backup.add(newNode.copy());
			}
			newNode.delete();
			newNode = null;
		}

		for (Vector b : beliefSet) {
			int a = - 1;
			double maxV = Double.NEGATIVE_INFINITY;
			for (int n = 0; n < fsc.size(); n++) {
				double v = Mtrx.dot(b, fsc.getNode(n).alpha);
				if (v > maxV) {
					maxV = v;
					a = fsc.getAction(n);
				}
			}
			for (int z = 0; z < nObservs; z++) {
				Vector b2 = pomdp.getNextBelief(b, a, z);
				double maxV2 = Double.NEGATIVE_INFINITY;
				for (int n = 0; n < backup.size(); n++) 
					maxV2 = Math.max(maxV2, Mtrx.dot(b2, backup.get(n).alpha));
				
				int dominantN = -1;
				double maxV1 = Double.NEGATIVE_INFINITY;			
				for (int n = 0; n < fsc.size(); n++) {
					double v = Mtrx.dot(b2, fsc.getNode(n).alpha);
					if (v > maxV1) {
						maxV1 = v;
						dominantN = n;
					}				
				}				
				if (maxV1 > maxV2) backup.add(fsc.getNode(dominantN).copy());
			}
		}
		
		double elapsedTime = CpuTime.getElapsedTime(startTime);
		if (bPrint)	System.out.printf("    Elapsed Time: %.2f sec\n", elapsedTime);
		return backup;
	}
		
	private FscNode[][] computeGammaAZ(FSC fsc, Vector b) {
		FscNode[][] gammaAZ = new FscNode[nActions][nObservs];
		for (int a = 0; a < nActions; a++) {
			for (int z = 0; z < nObservs; z++) {
				Vector b2 = pomdp.getNextBelief(b, a, z);
				double maxV = Double.NEGATIVE_INFINITY;
				int maxN = FscNode.NO_INFO;
				for (int n = 0; n < fsc.size(); n++) {
					double v = Mtrx.dot(b2, fsc.getNode(n).alpha);
					if (v > maxV) {
						maxN = n;
						maxV = v;
					}
				}
				gammaAZ[a][z] = fsc.getNode(maxN);
				b2 = null;
			}
		}
		return gammaAZ;
	}
	
	private FscNode pbSingleBackup(FSC fsc, Vector B) {
		FscNode newNode = new FscNode(nObservs, nStates, useSparse);		
		FscNode[][] gammaAZ= computeGammaAZ(fsc, B);
		double maxV = Double.NEGATIVE_INFINITY;
		Vector alpha = Mtrx.Vec(nStates, useSparse);
		
		for (int a = 0; a < nActions; a++) {
			alpha.zero();
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
						value += gamma * T * O * gammaAZ[a][z].alpha.get(s2);
					}
				}
				if (value != 0.0) alpha.set(s, value);
			}
			double v = Mtrx.dot(B, alpha);
			if (v > maxV) {
				maxV = v;
				newNode.act = a;
				newNode.alpha.zero();
				newNode.alpha.set(alpha);
			}
		}		
		for (int z = 0; z < nObservs; z++)
			newNode.nextNode[z] = gammaAZ[newNode.act][z].id;
		gammaAZ = null;
		alpha = null;
		return newNode;
	}
		
	private ArrayList<FscNode> fscTransformation(FSC fsc, ArrayList<FscNode> backup) {
		double startTime = CpuTime.getCurTime();
		if (bPrint)	System.out.printf("  Fsc Transformation : %d new nodes\n", backup.size());
		
		ArrayList<FscNode> newFsc = fsc.getNodesList();
		
		for (FscNode node : backup) {
			int sameNodeIndex = Mtrx.find(node, newFsc);
			if (sameNodeIndex != FscNode.NO_INFO) {
				newFsc.get(sameNodeIndex).alpha.set(node.alpha);
//				System.out.printf("@@@ n%d is equal to n%d\n", node.id, newFsc.get(sameNodeIndex).id);
				continue;
			}
			
			int dominateNodeId = FscNode.NO_INFO;
			for (int n2 = 0; n2 < newFsc.size(); n2++) {
				FscNode node2 = newFsc.get(n2);
				if (dominate(node, node2)) {
//					System.out.printf("@@@ n%d dominates n%d\n", node.id, node2.id);
					if (dominateNodeId == FscNode.NO_INFO) {
						node2.set(node);
						dominateNodeId = node2.id;
					}
					else {
						changeNextNodeId(newFsc, node2.id, dominateNodeId);
						changeNextNodeId(backup, node2.id, dominateNodeId);
						newFsc.remove(n2--);
					}				
				}
			}
			
			if (dominateNodeId == FscNode.NO_INFO) {
				newFsc.add(node.copy());
//				System.out.printf("@@@ n%d is inserted\n", node.id);
			}
		}		
		fsc.delete();
		double elapsedTime = CpuTime.getElapsedTime(startTime);
		if (bPrint)	System.out.printf("    Elapsed Time: %.2f sec\n", elapsedTime);
		return newFsc;
	}

	private boolean dominate(FscNode node, FscNode node2) {
		for (Vector b : beliefSet) {
			double v1 = Mtrx.dot(b, node.alpha);
			double v2 = Mtrx.dot(b, node2.alpha);
			if (v1 < v2) return false;
		}
		return true;
	}
	
	private void changeNextNodeId(ArrayList<FscNode> curSet, int id1, int id2) {
		for (FscNode node : curSet) {
			for (int z = 0; z < nObservs; z++)
				if (node.nextNode[z] == id1) node.nextNode[z] = id2;
		}
	}	

	private FSC pruneFsc(ArrayList<FscNode> newFsc, 
			ArrayList<FscNode> backup) {
		double T0 = CpuTime.getCurTime();
		if (bPrint)	System.out.println("  Policy Prunning");

		FSC fsc = new FSC(pomdp);
		ArrayList<FscNode> nonOptNodes = new ArrayList<FscNode>();
		for (FscNode node : newFsc) {
			if (Mtrx.find(node, backup) == FscNode.NO_INFO) nonOptNodes.add(node);
			else fsc.addNode(node.copy());
		}
		for (FscNode node : nonOptNodes) {
			boolean bReachable = false;
			for (FscNode node2 : newFsc) {
				if (node.id != node2.id) {
					for (int z = 0; z < nObservs; z++) {
						if (node2.nextNode[z] == node.id) {
							bReachable = true;
							break;
						}
					}
				}
				if (bReachable) break;
			}
			if (bReachable) fsc.addNode(node.copy());
//			else System.out.printf("@@@ n%d is pruned\n", node.id);
		}

		double T1 = CpuTime.getElapsedTime(T0);
		if (bPrint)	System.out.printf("    Elapsed Time: %.2f sec\n", T1);
		return fsc;
	}
		
////////////////////////////////////////////////////////////////////////////////
	
	private double calExpV(FSC fsc) {
		double sum = 0.0;
		for (Vector b : beliefSet)
			sum += fsc.calMaxV(b);
		return sum / beliefSet.size();
	}
	
	private void release(ArrayList<FscNode> set) {
		for (FscNode node : set)
			node.delete();
		set.clear();
		set = null;
	}
	
	public int getStartNode() 		{ return startNode; }
	public double getV0() 			{ return V0; }
	public double getElapsedTime() 	{ return totalElapsedTime; }
	public void print(FSC fsc) 		{ fsc.print(); }	
	public void print(FSC fsc, String filename) throws Exception { fsc.print(filename); }
}
