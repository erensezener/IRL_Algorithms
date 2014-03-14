package kr.ac.kaist.pomdp.data;


/**
 * Set of basis functions (or feature functions) for each problem.
 * 
 * The reward function is assumed to be linearly parameterized with the basis functions.
 * We prepare four set of basis functions for each problem: compact, non-compact, state-wise, and state-action-wise.
 * 
 * @see J. Choi and K. Kim, Inverse reinforcement learning in partially observable environments, JMLR 12, 2011.
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class BasisFunctions {	
	private PomdpProblem pomdp;		
	private int nStates;
	private int nActions;
	
	private int nBasis;
	private String probName;
	private String type;
	
	private int[][] trueRock = {
			{3, 1}, {2, 1}, {1, 3}, {1, 0}
	};
	private int[][] fakeRock = {
			{0, 3}, {2, 0}, {3, 3}, {1, 2}
	};
	
	public BasisFunctions(PomdpProblem _pomdp, String _probName, String _type) {
		pomdp = _pomdp;
		nStates = pomdp.nStates;
		nActions = pomdp.nActions;
		probName = _probName;
		type = _type;

		if (type.equals("S")) nBasis = nStates;
		else if (type.equals("SA")) nBasis = nStates * nActions;
		else if (type.equals("C") && probName.equals("heaven-hell")) nBasis = 5; 
		else if (type.equals("C") && probName.equals("RockSample_4_4")) nBasis = 13;
		else if (type.equals("NC") && probName.equals("RockSample_4_4")) nBasis = 19;
		else if (type.equals("C") && probName.equals("RockSample_4x3")) nBasis = 11;
		else if (type.equals("NC") && probName.equals("RockSample_4x3")) nBasis = 17;
		else nBasis = nStates * nActions;
	}	
	
	public double get(int i, int s, int a) {
		if (type.equals("S")) return fullS(i, s);
		else if (type.equals("SA")) return fullSA(i, s, a);
		else if (type.equals("C") && probName.equals("heaven-hell")) 
			return HeavenHell_C(i, s, a);
		else if (type.equals("C") && probName.equals("RockSample_4_4")) 
			return RockSample44_C(i, s, a);
		else if (type.equals("NC") && probName.equals("RockSample_4_4")) 
			return RockSample44_NC(i, s, a);
		else if (type.equals("C") && probName.equals("RockSample_4x3")) 
			return RockSample43_C(i, s, a);
		else if (type.equals("NC") && probName.equals("RockSample_4x3")) 
			return RockSample43_NC(i, s, a);
		else return fullSA(i, s, a);		
	}
	
	public int getNBasis() { return nBasis; }
	
	private double fullS(int i, int s) {
		if (i == s) return 1.0;
		else return 0.0;
	}
	
	private double fullSA(int i, int s, int a) {
		int x = i % nStates;
		int y = i / nStates;
		if (x == s && y == a) return 1.0;
		else return 0.0;
	}
	
	private double HeavenHell_C(int i, int s, int a) {
		// s = 4 (Heaven)
		if (i == 0 && s == 4) return 1;
		// s = 16 (Heaven)
		else if (i == 1 && s == 16) return 1;
		// s = 6 (Hell)
		else if (i == 2 && s == 6) return 1;
		// s = 14 (Hell)
		else if (i == 3 && s == 14) return 1;
		else if (i == 4 &&
				s != 4 && s != 16 && s != 6 && s != 14) return 1;
		else return 0;
	}
	
	private double RockSample44_C(int i, int s, int a) {
		int[] sData = parseRockSampleState(s);
		int x = sData[0];
		int y = sData[1];
		int r0 = sData[2];
		int r1 = sData[3];
		int r2 = sData[4];
		int r3 = sData[5];

		// actions: 0amn 1ame 2ams 3amw 4ac0 5ac1 6ac2 7ac3 8as
		// states: <x> <y> <r0> <r1> <r2> <r3>
				
		// x = 0, a = amw, reward = -100
		if (i == 0 && x == 0 && a == 3) return 1;
		// y = 0, a = ams, reward = -100
		else if (i == 1 && y == 0 && a == 2) return 1;
		// x = 3, a = ame, reward = 10
		else if (i == 2 && x == 3 && a == 1) return 1;
		// y = 3, a = amn, reward = -100
		else if (i == 3 && y == 3 && a == 0) return 1;

		// x = 3, y = 1, r0 = 1, a = as, reward = 10
		else if (i == 4 && a == 8 
				&& trueRock[0][0] == x && trueRock[0][1] == y && r0 == 1) return 1;
		// x = 3, y = 1, r0 = 0, a = as, reward = -10
		else if (i == 5 && a == 8 
				&& trueRock[0][0] == x && trueRock[0][1] == y && r0 == 0) return 1;
		// x = 2, y = 1, r1 = 1, a = as, reward = 10
		else if (i == 6 && a == 8 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r1 == 1) return 1;
		// x = 2, y = 1, r1 = 0, a = as, reward = -10
		else if (i == 7 && a == 8 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r1 == 0) return 1;
		// x = 1, y = 3, r2 = 1, a = as, reward = 10
		else if (i == 8 && a == 8 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r2 == 1) return 1;
		// x = 1, y = 3, r2 = 0, a = as, reward = -10
		else if (i == 9 && a == 8 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r2 == 0) return 1;
		// x = 1, y = 0, r3 = 1, a = as, reward = 10
		else if (i == 10 && a == 8 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r3 == 1) return 1;
		// x = 1, y = 0, r3 = 0, a = as, reward = -10
		else if (i == 11 && a == 8 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r3 == 0) return 1;
		// a = as, reward = -100
		else if (i == 12 && a == 8) {
			for (int j = 0; j < trueRock.length; j++)
				if (trueRock[j][0] == x && trueRock[j][1] == y) return 0;
			return 1;
		}		
		else return 0;
	}
	
	private double RockSample44_NC(int i, int s, int a) {
		int[] sData = parseRockSampleState(s);
		int x = sData[0];
		int y = sData[1];
		int r0 = sData[2];
		int r1 = sData[3];
		int r2 = sData[4];
		int r3 = sData[5];

		if (i == 0 && x == 0 && a == 3) return 1;
		else if (i == 1 && y == 0 && a == 2) return 1;
		else if (i == 2 && x == 3 && a == 1) return 1;
		else if (i == 3 && y == 3 && a == 0) return 1;
		
		else if (i == 4 && a == 8 
				&& trueRock[0][0] == x && trueRock[0][1] == y && r0 == 1) return 1;
		else if (i == 5 && a == 8 
				&& trueRock[0][0] == x && trueRock[0][1] == y && r0 == 0) return 1;
		else if (i == 6 && a == 8 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r1 == 1) return 1;
		else if (i == 7 && a == 8 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r1 == 0) return 1;
		else if (i == 8 && a == 8 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r2 == 1) return 1;
		else if (i == 9 && a == 8 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r2 == 0) return 1;
		else if (i == 10 && a == 8 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r3 == 1) return 1;
		else if (i == 11 && a == 8 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r3 == 0) return 1;
		
		else if (i == 12 && a == 8 
				&& fakeRock[0][0] == x && fakeRock[0][1] == y) return 1;
		else if (i == 13 && a == 8 
				&& fakeRock[1][0] == x && fakeRock[1][1] == y) return 1;
		else if (i == 14 && a == 8 
				&& fakeRock[2][0] == x && fakeRock[2][1] == y) return 1;
		else if (i == 15 && a == 8 
				&& fakeRock[3][0] == x && fakeRock[3][1] == y) return 1;
		
		else if (i == 16 && a == 8) {
			for (int j = 0; j < trueRock.length; j++)
				if (trueRock[j][0] == x && trueRock[j][1] == y) return 0;
			for (int j = 0; j < fakeRock.length; j++)
				if (fakeRock[j][0] == x && fakeRock[j][1] == y) return 0;
			return 1;
		}

		// a == ac0, 1, 2, 3
		else if (i == 17 && a >= 4 && a <= 7) return 1;
		
		// move on the map
		else if (i == 18 && a >= 0 && a <= 3
				&& !(x == 0 && a == 3)
				&& !(y == 0 && a == 2)
				&& !(x == 3 && a == 1)
				&& !(y == 3 && a == 0)) return 1;
		
		else return 0;		
	}
	
	private double RockSample43_C(int i, int s, int a) {
		int[] sData = parseRockSampleState2(s);
		int x = sData[0];
		int y = sData[1];
		int r0 = sData[2];
		int r1 = sData[3];
		int r2 = sData[4];

		// actions: 0amn 1ame 2ams 3amw 4ac0 5ac1 6ac2 7as
		// states: <x> <y> <r0> <r1> <r2>
				
		// x = 0, a = amw, reward = -100
		if (i == 0 && x == 0 && a == 3) return 1;
		// y = 0, a = ams, reward = -100
		else if (i == 1 && y == 0 && a == 2) return 1;
		// x = 3, a = ame, reward = 10
		else if (i == 2 && x == 3 && a == 1) return 1;
		// y = 3, a = amn, reward = -100
		else if (i == 3 && y == 3 && a == 0) return 1;

		// x = 3, y = 1, r0 = 1, a = as, reward = 10
		else if (i == 4 && a == 7 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r0 == 1) return 1;
		// x = 3, y = 1, r0 = 0, a = as, reward = -10
		else if (i == 5 && a == 7 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r0 == 0) return 1;
		// x = 2, y = 1, r1 = 1, a = as, reward = 10
		else if (i == 6 && a == 7 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r1 == 1) return 1;
		// x = 2, y = 1, r1 = 0, a = as, reward = -10
		else if (i == 7 && a == 7 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r1 == 0) return 1;
		// x = 1, y = 3, r2 = 1, a = as, reward = 10
		else if (i == 8 && a == 7 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r2 == 1) return 1;
		// x = 1, y = 3, r2 = 0, a = as, reward = -10
		else if (i == 9 && a == 7 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r2 == 0) return 1;
		// a = as, reward = -100
		else if (i == 10 && a == 7) {
			for (int j = 1; j < trueRock.length; j++)
				if (trueRock[j][0] == x && trueRock[j][1] == y) return 0;
			return 1;
		}		
		else return 0;
	}
	
	private double RockSample43_NC(int i, int s, int a) {
		int[] sData = parseRockSampleState2(s);
		int x = sData[0];
		int y = sData[1];
		int r0 = sData[2];
		int r1 = sData[3];
		int r2 = sData[4];

		if (i == 0 && x == 0 && a == 3) return 1;
		else if (i == 1 && y == 0 && a == 2) return 1;
		else if (i == 2 && x == 3 && a == 1) return 1;
		else if (i == 3 && y == 3 && a == 0) return 1;
		
		// x = 3, y = 1, r0 = 1, a = as, reward = 10
		else if (i == 4 && a == 7 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r0 == 1) return 1;
		// x = 3, y = 1, r0 = 0, a = as, reward = -10
		else if (i == 5 && a == 7 
				&& trueRock[1][0] == x && trueRock[1][1] == y && r0 == 0) return 1;
		// x = 2, y = 1, r1 = 1, a = as, reward = 10
		else if (i == 6 && a == 7 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r1 == 1) return 1;
		// x = 2, y = 1, r1 = 0, a = as, reward = -10
		else if (i == 7 && a == 7 
				&& trueRock[2][0] == x && trueRock[2][1] == y && r1 == 0) return 1;
		// x = 1, y = 3, r2 = 1, a = as, reward = 10
		else if (i == 8 && a == 7 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r2 == 1) return 1;
		// x = 1, y = 3, r2 = 0, a = as, reward = -10
		else if (i == 9 && a == 7 
				&& trueRock[3][0] == x && trueRock[3][1] == y && r2 == 0) return 1;
		
		else if (i == 10 && a == 7 
				&& fakeRock[0][0] == x && fakeRock[0][1] == y) return 1;
		else if (i == 11 && a == 7 
				&& fakeRock[1][0] == x && fakeRock[1][1] == y) return 1;
		else if (i == 12 && a == 7 
				&& fakeRock[2][0] == x && fakeRock[2][1] == y) return 1;
		else if (i == 13 && a == 7 
				&& fakeRock[3][0] == x && fakeRock[3][1] == y) return 1;
		
		else if (i == 14 && a == 7) {
			for (int j = 1; j < trueRock.length; j++)
				if (trueRock[j][0] == x && trueRock[j][1] == y) return 0;
			for (int j = 0; j < fakeRock.length; j++)
				if (fakeRock[j][0] == x && fakeRock[j][1] == y) return 0;
			return 1;
		}

		// a == ac0, 1, 2
		else if (i == 15 && a >= 4 && a <= 6) return 1;
		
		// move on the map
		else if (i == 16 && a >= 0 && a <= 3
				&& !(x == 0 && a == 3)
				&& !(y == 0 && a == 2)
				&& !(x == 3 && a == 1)
				&& !(y == 3 && a == 0)) return 1;
		
		else return 0;		
	}
	
	private int[] parseRockSampleState(int s) {
		String str = pomdp.states[s];
		int[] data = new int[] {-1, -1, -1, -1, -1, -1};
		if (!str.equals("st"))
			for (int i = 0; i < 6; i++)
				data[i] = Integer.parseInt(str.substring(i + 1, i + 2));
		return data;
	}
	
	private int[] parseRockSampleState2(int s) {
		String str = pomdp.states[s];
		int[] data = new int[] {-1, -1, -1, -1, -1};
		if (!str.equals("st"))
			for (int i = 0; i < 5; i++)
				data[i] = Integer.parseInt(str.substring(i + 1, i + 2));
		return data;
	}
}
