package kr.ac.kaist.pomdp.data;

import java.util.Iterator;
import java.util.Random;

import kr.ac.kaist.utils.Mtrx;

import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.SparseVector;

/**
 * Class for a POMDP problem
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class PomdpProblem {
	public boolean useSparse;	
	public int nStates;
	public int nActions;
	public int nObservations;
	public String values;
	public double gamma;
	
	public String[] states;	
	public String[] actions;
	public String[] observations;

	public Vector[][] T;
	public Vector[][] O;
	public Vector[] R;
	public Vector[][] R3;
	public Vector start;
	public double goodReward;
	public double badReward;
	public double RMAX;

	
	public PomdpProblem init(boolean uSparse) {
		useSparse = uSparse;
		if (R3 != null) simplifyReward();

		checkModel();
		
		goodReward = Double.NEGATIVE_INFINITY;
		for (int a = 0; a < this.nActions; a++)
			for (int s = 0; s < this.nStates; s++)
				goodReward = Math.max(goodReward, this.R[a].get(s));

		badReward = Double.POSITIVE_INFINITY;
		for (int a = 0; a < this.nActions; a++)
			for (int s = 0; s < this.nStates; s++)
				badReward = Math.min(badReward, this.R[a].get(s));
		
		RMAX = Math.max(Math.abs(badReward), Math.abs(goodReward));
		
		normalizeReward();

		return this;
	}
	
	private void normalizeReward() {
		for (int a = 0; a < this.nActions; a++) {
			for (int s = 0; s < this.nStates; s++) {
				double r = this.R[a].get(s);
				if (r != 0) {
					this.R[a].set(s, r / RMAX);
				}
			}
		}
	}

	public void delete() {
		for (int a = 0; a < nActions; a++) {
			for (int s = 0; s < nStates; s++) {
				if (T !=null) T[a][s] = null;
				if (O !=null) O[a][s] = null;
				if (R3 !=null) R3[a][s] = null;
			}
			R[a] = null;
		}
		if (states != null)
			for (int s = 0; s < nStates; s++)
				states[s] = null;
		states = null;
		
		if (actions != null) 
			for (int a = 0; a < nActions; a++)
				actions[a] = null;
		actions = null;
		
		if (observations != null)
			for (int z = 0; z < nObservations; z++)
				observations[z] = null;
		observations = null;
	}
	
	public void simplifyReward() {
		R = new Vector[nActions];
		for (int a = 0; a < this.nActions; a++) {
			R[a] = Mtrx.Vec(nStates, useSparse);
			for (int s = 0; s < this.nStates; s++) {
				double value = 0.0;
				Iterator<VectorEntry> itR = R3[a][s].iterator();
				while (itR.hasNext()) {
					VectorEntry veR = itR.next();
					int s2 = veR.index();
					value += T[a][s].get(s2) * R3[a][s].get(s2);
				}
				if (value != 0.0) R[a].set(s, value);
			}
			if (useSparse) ((SparseVector) R[a]).compact();
		}
	}
	
	public boolean checkModel() {
		boolean fail = false;
		for (int a = 0; a < nActions; a++) {
			for (int s = 0; s < nStates; s++) {
				double sum = Mtrx.sum(T[a][s]);
				if (sum != 1) {
//					System.out.printf("sum(T[s%d][a%d]) = %f !!!\n", s, a, sum);
					T[a][s] = Mtrx.normalize(T[a][s]);
					fail = true;
				}
				if (useSparse) ((SparseVector) T[a][s]).compact();
			}
		}
		
		for (int a = 0; a < nActions; a++) {
			for (int s = 0; s < nStates; s++) {
				double sum = Mtrx.sum(O[a][s]);
				if (sum != 1) {
//					System.out.printf("sum(O[s%d][a%d]) = %f !!!\n", s, a, sum);
					O[a][s] = Mtrx.normalize(O[a][s]);
					fail = true;
				}
				if (useSparse) ((SparseVector) O[a][s]).compact();
			}
		}
		
		for (int a = 0; a < nActions; a++)
			if (useSparse) ((SparseVector) R[a]).compact();
		
		double sum = Mtrx.sum(start);
		if (sum != 1) {
//			System.out.printf("sum of start belief = %f !!!\n", sum);
			start = Mtrx.normalize(start);
			fail = true;
		}
		if (useSparse) ((SparseVector) start).compact();
		
		return fail;
	}
		
	public int sampleState(Vector belief, Random rand) {
		int s = Mtrx.sample(belief, rand);
		return s;
	}
	
	public int sampleAction(Random rand) {
		int a = rand.nextInt(this.nActions); 
		return a;
	}
	
	public int sampleNextState(int state, int action, Random rand) {
		int s = Mtrx.sample(T[action][state], rand);
		return s;
	}
	
	public int sampleObserv(int action, int state, Random rand) {
		int obs = Mtrx.sample(O[action][state], rand);
		return obs;
	}	

	public Vector getNextBelief(Vector belief, int action, int z) {		
		Vector nextB = Mtrx.Vec(this.nStates, useSparse);
		double sum = 0.0;		
		for (int s2 = 0; s2 < nStates; s2++) {
			double pr = 0.0;
			double zPr = this.O[action][s2].get(z);
			if (zPr != 0.0) {
				Iterator<VectorEntry> itB = belief.iterator();
				while (itB.hasNext()) {
					VectorEntry veB = itB.next();
					int s = veB.index();
					double tPr = this.T[action][s].get(s2);
					pr += zPr * tPr * veB.get();
				}
			}
			if (pr != 0.0) {
				nextB.set(s2, pr);
				sum += pr;
			}
		}
		if (sum != 0.0) Mtrx.scale(nextB, 1.0 / sum);
		else Mtrx.uniform(nextB);
		return nextB;
	}
	
	public void printInfo() {
		System.out.println("=== POMDP ===================================================");
		System.out.println("discount: " + gamma);
		System.out.println("values: " + values);
		System.out.println("states:");
		for (int s = 0; s < nStates; s++) {
			System.out.print("[" + s);
			if (states != null) 
				System.out.print(":" + states[s]);
			System.out.print("] ");			
		}
		System.out.println("\nactions:");
		for (int a = 0; a < nActions; a++) {
			System.out.print("[" + a);
			if (actions != null) 
				System.out.print(":" + actions[a]);
			System.out.print("] ");			
		}
		System.out.println("\nobservations:");
		for (int z = 0; z < nObservations; z++) {
			System.out.print("[" + z);
			if (observations != null) 
				System.out.print(":" + observations[z]);
			System.out.print("] ");			
		}
		System.out.println("\nstart: ");
		for (int s = 0; s < nStates; s++) 
			System.out.print(start.get(s) + " ");		
		System.out.println();
		
		System.out.println("Transition Prob: ");
		for (int a = 0 ; a < nActions; a++) {
			System.out.println("  " + actions[a]);
			for (int s1 = 0; s1 < nStates; s1++) {
				System.out.print("    ");
				for (int s2 = 0; s2 < nStates; s2++) 
					System.out.printf("%f, ", T[a][s1].get(s2));
				System.out.println();
			}
		}
		
		System.out.println("Observation Prob: ");
		for (int a = 0 ; a < nActions; a++) {
			System.out.println("  " + actions[a]);
			for (int s = 0; s < nStates; s++) {
				System.out.print("    ");
				for (int obs = 0; obs < nObservations; obs++) 
					System.out.printf("%f, ", O[a][s].get(obs));
				System.out.println();
			}
		}
		
		System.out.println("Reward: ");
		for (int a = 0 ; a < nActions; a++) {
			System.out.println("  " + actions[a]);
			System.out.print("    ");
			for (int s = 0; s < nStates; s++)  
				System.out.printf("%f, ", R[a].get(s));
			System.out.println();
		}

		System.out.println();
	}
	
	public void printBriefInfo() {
		System.out.println("== Pomdp File ======================================================");
		System.out.printf("  states       : %d\n", nStates);
		System.out.printf("  actions      : %d\n", nActions);
		System.out.printf("  observations : %d\n", nObservations);
		System.out.printf("  gamma        : %f\n\n", gamma);
	}
	
	public void printSparseInfo() {
		System.out.println("== Pomdp File ======================================================");
		System.out.println("discount: " + gamma);
		System.out.println("values: " + values);
		System.out.println("states:");
		for (int s = 0; s < nStates; s++) {
			System.out.print("[" + s);
			if (states != null) 
				System.out.print(":" + states[s]);
			System.out.print("] ");			
		}
		System.out.println("\nactions:");
		for (int a = 0; a < nActions; a++) {
			System.out.print("[" + a);
			if (actions != null) 
				System.out.print(":" + actions[a]);
			System.out.print("] ");			
		}
		System.out.println("\nobservations:");
		for (int z = 0; z < nObservations; z++) {
			System.out.print("[" + z);
			if (observations != null) 
				System.out.print(":" + observations[z]);
			System.out.print("] ");			
		}
		System.out.println("\nstart: ");
		for (int s = 0; s < nStates; s++) 
			System.out.print(start.get(s) + " ");		
		System.out.println();
		
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				for (int s1 = 0; s1 < nStates; s1++) {
					if (T[a][s].get(s1) != 0) {
						System.out.print("T: ");
						if (actions != null) 
							System.out.print(actions[a]);
						else 
							System.out.print(a);
						System.out.print(": ");
						if (states != null) 
							System.out.println(states[s] + ": " + states[s1] + " " 
									+ T[a][s].get(s1));
						else 
							System.out.println(s + ": " + s1 + " " + T[a][s].get(s1));
					}                               
				}
			}
		}
		
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				for (int z = 0; z < nObservations; z++) {
					if (O[a][s].get(z) != 0) {
						System.out.print("O: ");
						if (actions != null) 
							System.out.print(actions[a]);
						else 
							System.out.print(a);
						System.out.print(": ");
						if (states != null) 
							System.out.println(states[s] + ": " + observations[z] + " " 
									+ O[a][s].get(z));
						else 
							System.out.println(s + ": " + z + " " + O[a][s].get(z));
					}                               
				}
			}
		}
		
		for (int s = 0; s < nStates; s++) {
			for (int a = 0; a < nActions; a++) {
				if (R[a].get(s) != 0) {
					System.out.print("R: ");
						if (actions != null) 
							System.out.print(actions[a]);
						else 
							System.out.print(a);
						System.out.print(": ");
						if (states != null) 
							System.out.println(states[s] + " " + R[a].get(s));
						else 
							System.out.println(s + " " + R[a].get(s));
				}
			}
		}
		System.out.println();
	}
}
