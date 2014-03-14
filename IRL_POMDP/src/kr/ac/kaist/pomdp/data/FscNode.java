package kr.ac.kaist.pomdp.data;

import kr.ac.kaist.utils.Mtrx;
import no.uib.cipr.matrix.Vector;

/**
 * Class for a node of finite state controller (FSC)
 * This is used for representing a policy of POMDP
 * 
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class FscNode {
	public static int NO_INFO = Integer.MIN_VALUE;
	
	private boolean useSparse;
	public int id;
	public int act;
	public int[] nextNode;
	public Vector alpha;
	
	public FscNode(int nObservs, int nStates, boolean useSparse) {
		this.id = NO_INFO;
		this.nextNode = new int[nObservs];
		this.useSparse = useSparse;
		this.alpha = Mtrx.Vec(nStates, useSparse);
	}
	
	public FscNode(int id, int nObservs, int nStates, boolean useSparse) {
		this.id = id;
		this.nextNode = new int[nObservs];
		this.useSparse = useSparse;
		this.alpha = Mtrx.Vec(nStates, useSparse);
	}

	public FscNode copy() {
		FscNode n2 = new FscNode(this.id, this.nextNode.length, this.alpha.size(), this.useSparse);
		n2.act = this.act;
		for (int i = 0; i < this.nextNode.length; i++)
			n2.nextNode[i] = this.nextNode[i];
		n2.alpha.set(this.alpha);
		return n2;
	}
	
	public boolean equalAlpha(FscNode n2) {
		return Mtrx.equal(this.alpha, n2.alpha);
	}
	
	public boolean equals(FscNode n2) {
		if (n2 == null) return false;
		if (this.act != n2.act) return false;
		if (this.nextNode.length != n2.nextNode.length) return false;
		for (int i = 0; i < this.nextNode.length; i++)
			if (this.nextNode[i] != n2.nextNode[i]) return false;
		return true;
	}
	
	public void setAlpha(double[] v) {
		for (int i = 0; i < this.alpha.size(); i++)
			if (v[i] != 0.0) this.alpha.set(i, v[i]);
	}
	
	public void set(FscNode n2) {
		this.act = n2.act;
		this.alpha.set(n2.alpha);
		for (int i = 0; i < this.nextNode.length; i++)
			this.nextNode[i] = n2.nextNode[i];
	}

	public void delete() {
		nextNode = null;
		alpha = null;
	}
}
