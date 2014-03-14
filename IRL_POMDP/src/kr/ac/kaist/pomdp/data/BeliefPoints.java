package kr.ac.kaist.pomdp.data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

import kr.ac.kaist.utils.Mtrx;


import no.uib.cipr.matrix.Vector;

/**
 * Class for some functions handling belief points
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class BeliefPoints {

	public static ArrayList<Vector> initBeliefs(PomdpProblem pomdp, ArrayList<Vector> set, 
			int nBeliefs, int maxRestart, double minDistBeliefs, Random rand) {
		ArrayList<Vector> beliefSet = new ArrayList<Vector>();
		for (Vector b : set) beliefSet.add(b.copy());
		int restartCnt = 0;
		while (beliefSet.size() < nBeliefs && restartCnt < maxRestart) {
			//beliefSet.add(randBelief(pomdp.nStates, rand));
			ArrayList<Vector> newB = expandSSEA(pomdp, beliefSet, minDistBeliefs, rand);
			if (beliefSet.size() == newB.size()) restartCnt++;
			else restartCnt = 0;
			beliefSet.clear();
			beliefSet.addAll(newB);
			newB.clear();
			newB = null;
		}
		return beliefSet;
	}
	
	public static ArrayList<Vector> initBeliefs(PomdpProblem pomdp, 
			int nBeliefs, int maxRestart, double minDistBeliefs, Random rand) {
		ArrayList<Vector> beliefSet = new ArrayList<Vector>();
		beliefSet.add(pomdp.start.copy());
		int restartCnt = 0;
		while (beliefSet.size() < nBeliefs && restartCnt < maxRestart) {
			//beliefSet.add(randBelief(pomdp.nStates, rand));
			ArrayList<Vector> newB = expandSSEA(pomdp, beliefSet, minDistBeliefs, rand);
			if (beliefSet.size() == newB.size()) restartCnt++;
			else restartCnt = 0;
			beliefSet.clear();
			beliefSet.addAll(newB);
			newB.clear();
			newB = null;
		}
		return beliefSet;
	}
	
	private static Vector randBelief(int nStates, Random rand) {
		Vector b = Mtrx.Vec(nStates, true);
		for (int i = 0; i < nStates; i++) 
			b.set(i, rand.nextDouble());
		b = Mtrx.normalize(b);
		return b;
	}
	
	public static ArrayList<Vector> expandSSEA(PomdpProblem pomdp, 
			ArrayList<Vector> beliefSet, double minDistBeliefs, Random rand) {
		ArrayList<Vector> newBeliefSet = new ArrayList<Vector>();
		Iterator<Vector> it = beliefSet.iterator();
		while (it.hasNext()) 
			newBeliefSet.add(it.next().copy());
		
		it = beliefSet.iterator();
		while (it.hasNext()) {
			double maxDist = Double.NEGATIVE_INFINITY;
			int maxDistA = -1;
			
			Vector b = it.next();
			Vector[] bA = new Vector[pomdp.nActions];
			for (int a = 0; a < pomdp.nActions; a++) {
				int s = pomdp.sampleState(b, rand);
				int s2 = pomdp.sampleNextState(s, a, rand);
				int z = pomdp.sampleObserv(a, s2, rand);
				bA[a] = pomdp.getNextBelief(b, a, z);
				
				double dist = Double.POSITIVE_INFINITY;
				Iterator<Vector> it2 = newBeliefSet.iterator();
				while (it2.hasNext()) {
					Vector b2 = it2.next();
					dist = Math.min(dist, Mtrx.calL2Dist(bA[a], b2));
				}
				if (dist > maxDist) {
					maxDist = dist;
					maxDistA = a;
				}
			}
			if (maxDist > minDistBeliefs) newBeliefSet.add(bA[maxDistA].copy());
			for (int a = 0; a < pomdp.nActions; a++) bA[a] = null;
			bA = null;
		}
		return newBeliefSet;
	}
}
