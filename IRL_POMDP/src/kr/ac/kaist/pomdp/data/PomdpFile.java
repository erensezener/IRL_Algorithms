package kr.ac.kaist.pomdp.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import kr.ac.kaist.pomdp.data.PomdpProblem;
import kr.ac.kaist.utils.Mtrx;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;

/**
 * Class for parsing POMDP files
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class PomdpFile
{
	public static PomdpProblem read(String filename, boolean useSparse) 
	throws Exception {
		ArrayList<String> file = new ArrayList<String>();
		String buffer = null;

		BufferedReader reader = new BufferedReader(new FileReader(filename));
		// read a line. do not store comments and empty lines
		while ((buffer = reader.readLine()) != null) {
			int length = buffer.length();
			if (length <= 0) continue;
			
			int comment = buffer.indexOf("#");
			if (comment > 0)
				file.add(buffer.substring(0, comment - 1));
			else if (comment == 0)
				continue;
			else 
				file.add(buffer);
		}
		reader.close();
		
		PomdpProblem pomdp = processPreamble(file, useSparse);

		// process each line
		for (int iLine = 0; iLine < file.size(); iLine++) {
			if (file.get(iLine).length() <=0 ) 
				continue;
			
			switch (file.get(iLine).charAt(0)) {
			case 'T':
				if (file.get(iLine).indexOf(':') >= 0)
					pomdp = processTransition(pomdp, file, iLine);
				break;
			case 'R':
				if (file.get(iLine).indexOf(':') >= 0)
					pomdp = processReward(pomdp, file, iLine);
				break;
			case 'O':
				if (file.get(iLine).indexOf(':') >= 0)
					pomdp = processObservation(pomdp, file, iLine);
				break;
			case 's':
				if (file.get(iLine).substring(0, 6).equals("start:"))
					pomdp = processStart(pomdp, file, iLine);
				break;
			}
		}
		
		// check start
		if (Mtrx.sum(pomdp.start) == 0)
			for (int s = 0; s < pomdp.nStates; s++) 
				pomdp.start.set(s, 1.0 / pomdp.nStates);

		return pomdp.init(useSparse);
	}
	
	protected static PomdpProblem processPreamble(ArrayList<String> file, 
			boolean useSparse) {
		PomdpProblem pomdp = new PomdpProblem();

		ArrayList states = getNumberAndMembers(file, "states:");
		pomdp.nStates = ((Integer) states.get(0)).intValue();
		pomdp.states = (String[]) states.get(1);

		ArrayList actions = getNumberAndMembers(file, "actions:");
		pomdp.nActions = ((Integer) actions.get(0)).intValue();
		pomdp.actions = (String[]) actions.get(1);

		ArrayList obs = getNumberAndMembers(file, "observations:");
		pomdp.nObservations = ((Integer) obs.get(0)).intValue();
		pomdp.observations = (String[]) obs.get(1);

		for (int i = 0; i < file.size(); i++) {
			if (file.get(i).contains("discount:")) {
				String str = file.get(i).split("discount:\\s*")[1];
				pomdp.gamma = Double.parseDouble(str);
				break;
			}
		}
		
		for (int i = 0; i < file.size(); i++) {
			if (file.get(i).contains("values:")) {
				pomdp.values = file.get(i).split("values:\\s*")[1];
				break;
			}
		}

		pomdp.start = Mtrx.Vec(pomdp.nStates, useSparse);		
		pomdp.T = new Vector[pomdp.nActions][pomdp.nStates];
		pomdp.O = new Vector[pomdp.nActions][pomdp.nStates];
		pomdp.R3 = new Vector[pomdp.nActions][pomdp.nStates];
		//pomdp.R = new Vector[pomdp.nActions];
		for (int a = 0; a < pomdp.nActions; a++) {
			for (int s = 0; s < pomdp.nStates; s++) {
				pomdp.T[a][s] = Mtrx.Vec(pomdp.nStates, useSparse);
				pomdp.O[a][s] = Mtrx.Vec(pomdp.nStates, useSparse);
				pomdp.R3[a][s] = Mtrx.Vec(pomdp.nStates, useSparse);
			}				
			//pomdp.R[a] = MtjUtil.Vec(pomdp.nStates, useSparse);
		}

		return pomdp;
	}

	protected static ArrayList getNumberAndMembers(ArrayList<String> file, String baseStr) {
		ArrayList result = new ArrayList(2);
		String str = null;
		for (int i = 0; i < file.size(); i++) {
			if (file.get(i).contains(baseStr)) {
				str = file.get(i);
				break;
			}
		}

		Matcher mat = Pattern.compile(baseStr + "\\s*(\\d+)\\s*").matcher(str);
		if (!mat.matches()) {
			// catch 'X: <list of X>' where X = {states, actions, observations}
			// first strip baseStr
			String str2 = str.split(baseStr + "\\s*")[1];
			// see if there are more members on the next line
			String[] elements = str2.split("\\s+");
			result.add(0, elements.length);
			result.add(1, elements);
		}
		else {
			// catch 'X: %d' where X = {states, actions, observations}
			result.add(0, new Integer(mat.group(1)));
			result.add(1, null);
		}

		return result;
	}

	protected static PomdpProblem processTransition(PomdpProblem pomdp, 
			ArrayList<String> file, int iLine) {
		double prob = 0;
		String str = file.get(iLine);
		if (str.split(":").length == 4) {
			// catch 'T: <action> : <start-state> : <end-state> <prob>'
			String patStr = "T\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s+([\\d\\.]+)";
			Pattern pat = Pattern.compile(patStr);
			Matcher mat = pat.matcher(str);

			if (mat.matches())
				prob = Double.parseDouble(mat.group(4));
			else {
				// catch 'T: <action> : <start-state> : <end-state> \n <prob>'
				patStr = "T\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*";
				pat = Pattern.compile(patStr);
				mat = pat.matcher(str);
				mat.find();				
				prob = parseNextLine(file, iLine + 1, 1, 1, pomdp.useSparse).get(0, 0);
			}
			
			ArrayList<Integer> action = expandAction(pomdp, mat.group(1));
			ArrayList<Integer> from = expandState(pomdp, mat.group(2));
			ArrayList<Integer> to = expandState(pomdp, mat.group(3));
			for (int a = 0; a < action.size(); a++)
				for (int s = 0; s < from.size(); s++)
					for (int ns = 0; ns < to.size(); ns++)
						if (prob != 0.0)
							pomdp.T[action.get(a)][from.get(s)].set(to.get(ns), prob);
								
		}
		else if (str.split(":").length == 3) {
			// catch 'T: <action> : <start-state>'
			Pattern pat = Pattern.compile("T\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)");
			Matcher mat = pat.matcher(str);
			mat.find();

			ArrayList<Integer> action = expandAction(pomdp, mat.group(1));
			ArrayList<Integer> from = expandState(pomdp, mat.group(2));
			Matrix probs = parseNextLine(file, iLine + 1, 1, pomdp.nStates, pomdp.useSparse);

			for (int a = 0; a < action.size(); a++)
				for (int s = 0; s < from.size(); s++)
					for (int ns = 0; ns < pomdp.nStates; ns++) 
						if (probs.get(0, ns) != 0.0)
							pomdp.T[action.get(a)][from.get(s)].set(ns, probs.get(0, ns));
		}
		else {
			// catch 'T: <action>'
			Pattern pat = Pattern.compile("T\\s*:\\s*(\\S+)\\s*");
			Matcher mat = pat.matcher(str);
			mat.find();

			ArrayList<Integer> action = expandAction(pomdp, mat.group(1));
			Matrix values = parseNextLine(file, iLine + 1, 
					pomdp.nStates, pomdp.nStates, pomdp.useSparse);

			for (int a = 0; a < action.size(); a++) {
				for (int s = 0; s < pomdp.nStates; s++) {
					for (int ns = 0; ns < pomdp.nStates; ns++) {
						if (values.get(s, ns) != 0.0)
							pomdp.T[action.get(a)][s].set(ns, values.get(s, ns));
					}
				}
			}
		}

		return pomdp;
	}

	protected static PomdpProblem processReward(PomdpProblem pomdp, 
			ArrayList<String> file,	int iLine) {
		String str = file.get(iLine);
		double reward = 0;
		if (str.split(":").length == 5) {
			// catch 'R: <action> : <start-state> : <end-state> : <observation> <reward>'
			String patStr = 
				"R\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s+([-\\d\\.]+)";
			Pattern pat = Pattern.compile(patStr);
			Matcher mat = pat.matcher(str);
			if (mat.find())
				reward = Double.parseDouble(mat.group(5));
			else {
				// probably the reward is on the next line
				// catch 'R: <action> : <start-state> : <end-state> : <observation> \n <reward>'
				patStr = "R\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*";
				pat = Pattern.compile(patStr);
				mat = pat.matcher(str);
				mat.find();
				reward = parseNextLine(file, iLine + 1, 1, 1, pomdp.useSparse).get(0, 0);
			}

			ArrayList<Integer> action = expandAction(pomdp, mat.group(1));
			ArrayList<Integer> from = expandState(pomdp, mat.group(2));
			ArrayList<Integer> to = expandState(pomdp, mat.group(3));

			// we ignore the observation
			for (int a = 0; a < action.size(); a++)
				for (int s = 0; s < from.size(); s++)
					for (int ns = 0; ns < to.size(); ns++) 
						if (reward != 0.0)
							pomdp.R3[action.get(a)][from.get(s)].set(to.get(ns), reward);
			
//			for (int a = 0; a < action.size(); a++)
//				for (int s = 0; s < from.size(); s++)
//					pomdp.R[action.get(a)].set(from.get(s), reward);
		}
		else
			System.out.println("ERROR in parsing reward");

		return pomdp;
	}

	protected static PomdpProblem processObservation(PomdpProblem pomdp, 
			ArrayList<String> file,	int iLine) {
		String str = file.get(iLine);
		double prob = 0;
		if (str.split(":").length == 4) {
			// catch 'O: <action> : <state> : <observation> <prob>'
			String patStr = "O\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s+([\\d\\.]+)";
			Pattern pat = Pattern.compile(patStr);
			Matcher mat = pat.matcher(str);

			if (mat.find())
				prob = Double.parseDouble(mat.group(4));
			else {
				// probably the prob is on the next line
				// catch 'O: <action> : <state> : <observation> \n <prob>'
				pat = Pattern.compile("O\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)\\s*");
				mat = pat.matcher(str);
				mat.find();
				prob = parseNextLine(file, iLine + 1, 1, 1, pomdp.useSparse).get(0, 0);
			}

			ArrayList<Integer> action = expandAction(pomdp, mat.group(1));
			ArrayList<Integer> to = expandState(pomdp, mat.group(2));
			ArrayList<Integer> observation = expandObservation(pomdp, mat.group(3));

			for(int a = 0; a < action.size(); a++) 
				if (prob != 0.0)
					pomdp.O[action.get(a)][to.get(0)].set(observation.get(0), prob);
		}
		else if (str.split(":").length == 3) {
			// catch 'O: <action> : <state>'
			Pattern pat = Pattern.compile("O\\s*:\\s*(\\S+)\\s*:\\s*(\\S+)");
			Matcher mat = pat.matcher(str);
			mat.find();

			ArrayList<Integer> action = expandAction(pomdp, mat.group(1));
			ArrayList<Integer> to = expandState(pomdp, mat.group(2));
			Matrix probs = parseNextLine(file, iLine + 1, 1, pomdp.nObservations, pomdp.useSparse);

			for (int a = 0; a < action.size(); a++)
				for (int s = 0; s < to.size(); s++)
					for (int z = 0; z < pomdp.nObservations; z++) 
						if (probs.get(0, z) != 0.0)
							pomdp.O[action.get(a)][to.get(s)].set(z, probs.get(0, z));
					
		}
		else if (str.split(":").length == 2) {
			// catch 'O: <action>
			Pattern pat = Pattern.compile("O\\s*:\\s*(\\S+)\\s*");
			Matcher mat = pat.matcher(str);
			mat.find();

			ArrayList<Integer> action = expandAction(pomdp, mat.group(1));
			Matrix values = parseNextLine(file, iLine + 1, 
					pomdp.nStates, pomdp.nObservations, pomdp.useSparse);
			for (int a = 0; a < action.size(); a++) 
				for (int s = 0; s < pomdp.nStates; s++) 
					for (int z = 0; z < pomdp.nObservations; z++) 
						if (values.get(s, z) != 0.0)
							pomdp.O[action.get(a)][s].set(z, values.get(s, z));
		}
		return pomdp;
	}

	protected static PomdpProblem processStart(PomdpProblem pomdp, 
			ArrayList<String> file,	int iLine) {
		String str = file.get(iLine);
		Pattern pat = Pattern.compile("([-\\d\\.]+)");
		Matcher mat = pat.matcher(file.get(iLine));
		int k;
		for (k = 0; k < pomdp.nStates; k++) {
			if (mat.find())
				pomdp.start.set(k, Double.parseDouble(str.substring(mat.start(), mat.end())));
			else
				break;
		}
		if (k < pomdp.nStates) {
			Matrix values = parseNextLine(file, iLine + 1, 1, pomdp.nStates, pomdp.useSparse);
			for (k = 0; k < pomdp.nStates; k++)
				if (values.get(0, k) != 0.0)
					pomdp.start.set(k, values.get(0, k));
		}
		return pomdp;
	}
	
	protected static Matrix parseNextLine(ArrayList<String> file, int iLine, 
			int nrRows, int nrCols, boolean useSparse) {
		Matrix values = Mtrx.Mat(nrRows, nrCols, useSparse);

		if (file.get(iLine).startsWith("uniform")) {
			for (int row = 0; row < nrRows; row++)
				for (int col = 0; col < nrCols; col++)
					values.set(row, col, 1.0 / nrCols);
		}
		else if (file.get(iLine).startsWith("identity")) {
			for (int row = 0; row < nrRows; row++)
				for (int col = 0; col < nrCols; col++)
					values.set(row, col, (row == col) ? 1.0 : 0.0);
		} 
		else {			
			String patStr = new String();
			for (int c = 0; c < nrCols; c++)
				patStr = patStr.concat("([\\d\\.]+)\\s*");
			Pattern pat = Pattern.compile(patStr);
			Matcher mat = null;
			for (int r = 0; r < nrRows; r++) {
				mat = pat.matcher(file.get(iLine + r));
				if (mat.find())
					for (int c = 0; c < nrCols; c++)
						values.set(r, c, Double.parseDouble(mat.group(c + 1)));
				else
					System.out.printf("ERROR in parsing next line: %d %d %d\n", r, nrRows, nrCols);
					//System.out.println("ERROR in parsing next line");
			}
		}
		return values;
	}

	protected static ArrayList<Integer> expandAction(PomdpProblem pomdp, String c) {
		return expandString(c, pomdp.nActions, pomdp.actions);
	}

	protected static ArrayList<Integer> expandState(PomdpProblem pomdp, String c) {
		return expandString(c, pomdp.nStates, pomdp.states);
	}

	protected static ArrayList<Integer> expandObservation(PomdpProblem pomdp, String c) {
		return expandString(c, pomdp.nObservations, pomdp.observations);
	}

	protected static ArrayList<Integer> expandString(String c, int nr, String[] members) {
		ArrayList<Integer> idList = new ArrayList<Integer>();
		if (c.equals("*"))
			for (int i = 0; i < nr; i++)
				idList.add(new Integer(i));
		else {
			if (members != null) {
				for (int i = 0; i < nr; i++)
					if (members[i].equals(c))
						idList.add(new Integer(i));
			}
			// apparently c is a numbered state
			if (idList.size() == 0) 
				idList.add(new Integer(c));
		}
		return idList;
	}
}
