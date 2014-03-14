package kr.ac.kaist.pomdp.data;

import java.io.BufferedWriter;
import java.io.FileWriter;

import kr.ac.kaist.utils.Mtrx;

import no.uib.cipr.matrix.Vector;

/**
 * Class for generating RockSample problems
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class GenProblems {
	private static String fname = "./problems/RockSample_4x3.pomdp";
	private static PomdpProblem pomdp;		
	private static int nStates;
	private static int nActions;
	private static int nObservs;
	private static boolean useSparse;


	public static void modifyRockSample2(PomdpProblem _pomdp) throws Exception {
		pomdp = _pomdp;
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		nObservs = pomdp.nObservations;
		useSparse = pomdp.useSparse;

		Vector[][] T = new Vector[nActions][nStates];
		for (int a = 0; a < nActions; a++) {
			for (int s = 0; s < nStates - 1; s++) {
				T[a][s] = Mtrx.Vec(nStates, useSparse);
				for (int s2 = 0; s2 < nStates; s2++) {
					double pr = pomdp.T[a][s].get(s2);
					if (pr != 0) T[a][s].set(s2, pr);
				}
			}
			int st = nStates - 1;
			T[a][st] = Mtrx.Vec(nStates, useSparse);
			for (int s0 = 0; s0 < nStates; s0++) {
				double b0 = pomdp.start.get(s0);
				if (b0 != 0) T[a][st].set(s0, b0);
			}
		}
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(fname));
		bw.write("# This is the POMDP file for the non-epsodic Rock Sample [4,3] problem.\n");
		bw.write("# This is generated from the original Rock Sample [4,3] problem,\n");
		bw.write("# which is built by Trey Smith.\n");
		bw.write("# This is written by Jaedeug Choi.\n\n");

		bw.write("discount: " + pomdp.gamma + "\n");
		bw.write("values: reward\n\n");

		bw.write("states: ");
		for (int i = 0; i < nStates; i++)
			bw.write(pomdp.states[i] + " ");
		bw.write("\n\n");

		bw.write("actions: ");
		for (int i = 0; i < nActions; i++)
			bw.write(pomdp.actions[i] + " ");
		bw.write("\n\n");

		bw.write("observations: ");
		for (int i = 0; i < nObservs; i++)
			bw.write(pomdp.observations[i] + " ");
		bw.write("\n\n");

		bw.write("start: ");
		for (int i = 0; i < nStates; i++)
			bw.write(pomdp.start.get(i) + " ");
		bw.write("\n\n");

		for (int s = 0; s < nStates; s++) {
			String strS1 = pomdp.states[s];
			for (int a = 0; a < nActions; a++) {
				String strA = pomdp.actions[a];
				if (strA.length() < 3) strA = strA.concat(" ");
				for (int s2 = 0; s2 < nStates; s2++) {
					String strS2 = pomdp.states[s2];
					double pr = T[a][s].get(s2);
					if (pr != 0) {
						bw.write("T: " + strA + " : ");
						bw.write(strS1 + "\t: ");
						bw.write(strS2 + "\t ");
						bw.write(pr + "\n");
					}
				}
				for (int z = 0; z < nObservs; z++) {
					String strZ = pomdp.observations[z];
					if (strZ.length() < 4) strZ = strZ.concat("   ");
					else strZ = strZ.concat("  ");
					double pr = pomdp.O[a][s].get(z);
					if (pr != 0) {
						bw.write("O: " + strA + " : ");
						bw.write(strS1 + "\t: ");
						bw.write(strZ + "\t ");
						bw.write(pr + "\n");						
					}
				}
				double r = pomdp.R[a].get(s);
				if (r != 0) {
					bw.write("R: " + strA + " : ");
					bw.write(strS1 + "\t: *\t : *\t ");
					bw.write(r + "\n");					
				}
				bw.write("\n");
			}
		}
		bw.write("\n");
		bw.close();
	}

	public static void modifyRockSample(PomdpProblem _pomdp) throws Exception {
		pomdp = _pomdp;
		nStates = pomdp.nStates - 1;
		nActions = pomdp.nActions;
		nObservs = pomdp.nObservations;
		useSparse = pomdp.useSparse;

		Vector[][] T = new Vector[nActions][nStates];
		for (int a = 0; a < nActions; a++) {
			for (int s = 0; s < nStates; s++) {
				T[a][s] = Mtrx.Vec(nStates, useSparse);
				for (int s2 = 0; s2 < nStates + 1; s2++) {
					double pr = pomdp.T[a][s].get(s2);
					if (pr != 0) {
						if (isEnd(s2)) {
							for (int s0 = 0; s0 < nStates; s0++) {
								double b0 = pomdp.start.get(s0);
								if (b0 != 0) T[a][s].set(s0, b0);
							}
						}
						else T[a][s].set(s2, pr);
					}
				}
			}
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(fname));
		bw.write("# This is the POMDP file for the non-epsodic Rock Sample [4,3] problem.\n");
		bw.write("# This is generated from the original Rock Sample [4,3] problem,\n");
		bw.write("# which is built by Trey Smith.\n");
		bw.write("# This is written by Jaedeug Choi.\n\n");

		bw.write("discount: " + pomdp.gamma + "\n");
		bw.write("values: reward\n\n");

		bw.write("states: ");
		for (int i = 0; i < nStates; i++)
			bw.write(pomdp.states[i] + " ");
		bw.write("\n\n");

		bw.write("actions: ");
		for (int i = 0; i < nActions; i++)
			bw.write(pomdp.actions[i] + " ");
		bw.write("\n\n");

		bw.write("observations: ");
		for (int i = 0; i < nObservs; i++)
			bw.write(pomdp.observations[i] + " ");
		bw.write("\n\n");

		bw.write("start: ");
		for (int i = 0; i < nStates; i++)
			bw.write(pomdp.start.get(i) + " ");
		bw.write("\n\n");

		for (int s = 0; s < nStates; s++) {
			String strS1 = pomdp.states[s];
			for (int a = 0; a < nActions; a++) {
				String strA = pomdp.actions[a];
				if (strA.length() < 3) strA = strA.concat(" ");
				for (int s2 = 0; s2 < nStates; s2++) {
					String strS2 = pomdp.states[s2];
					double pr = T[a][s].get(s2);
					if (pr != 0) {
						bw.write("T: " + strA + " : ");
						bw.write(strS1 + "\t: ");
						bw.write(strS2 + "\t ");
						bw.write(pr + "\n");
					}
				}
				for (int z = 0; z < nObservs; z++) {
					String strZ = pomdp.observations[z];
					if (strZ.length() < 4) strZ = strZ.concat("   ");
					else strZ = strZ.concat("  ");
					double pr = pomdp.O[a][s].get(z);
					if (pr != 0) {
						bw.write("O: " + strA + " : ");
						bw.write(strS1 + "\t: ");
						bw.write(strZ + "\t ");
						bw.write(pr + "\n");						
					}
				}
				double r = pomdp.R[a].get(s);
				if (r != 0) {
					bw.write("R: " + strA + " : ");
					bw.write(strS1 + "\t: *\t : *\t ");
					bw.write(r + "\n");					
				}
				bw.write("\n");
			}
		}
		bw.write("\n");
		bw.close();
	}

	private static boolean isEnd(int s) {
		if (s == nStates) return true;
		else return false;
	}
}
