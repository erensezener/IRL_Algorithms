package kr.ac.kaist.utils;

import java.util.ArrayList;
import java.util.Random;
import no.uib.cipr.matrix.sparse.*;
import no.uib.cipr.matrix.*;

import java.util.Iterator;

import kr.ac.kaist.pomdp.data.FscNode;

/**
 * Class for manipulating vectors and matricies
 *  
 * @author Jaedeug Choi (jdchoi@ai.kaist.ac.kr)
 *
 */
public class Mtrx {
	public static double PRECISION = 1e-10;

	public static Vector Vec(int dim, boolean useSparse) {
		Vector v;
		if (useSparse) v = new SparseVector(dim);
		else v = new DenseVector(dim);
		return v;
	}
	
	public static Matrix Mat(int nRows, int nCols, boolean useSparse) {
		Matrix m;
		if (useSparse) m = new FlexCompColMatrix(nRows, nCols);
		else m = new DenseMatrix(nRows, nCols);
		return m;
	}
	
	public static Iterator<VectorEntry> Iter(Vector v) {
		Iterator<VectorEntry> it;
		if (v instanceof SparseVector) it = ((SparseVector) v).iterator();
		else it = ((DenseVector) v).iterator();
		return it;
	}
	
	public static Iterator<MatrixEntry> Iter(Matrix m) {
		Iterator<MatrixEntry> it;
		if (m instanceof FlexCompColMatrix) it = ((FlexCompColMatrix) m).iterator();
		else it = ((DenseMatrix) m).iterator();
		return it;
	}
		
	public static int sample(Vector v, Random rand) {
		v = Mtrx.normalize(v);
		double prob = rand.nextDouble();
		double cumsum = 0.0;
		Iterator<VectorEntry> it = Iter(v);
		while (it.hasNext()) {
			VectorEntry ve = it.next();
			cumsum += ve.get();
			if (prob <= cumsum) return ve.index();
		}
		return -1;
	}

	public static Vector multiply(Vector a, Vector b, Vector result) {
		result.zero();
		Iterator<VectorEntry> it = Iter(a);
		while (it.hasNext()) {
			VectorEntry ve = it.next();
			int i = ve.index();
			double x = ve.get() * b.get(i);
			if (x != 0.0) result.set(i, x);
		}
		return result;
	}
	
	// this can be improved
	public static double dot(Vector a, Vector b) {
		double result = 0.0;
		if (a instanceof SparseVector && b instanceof SparseVector) {
			int na = ((SparseVector) a).getUsed();
			int nb = ((SparseVector) b).getUsed();
			if (na < nb) {
				Iterator<VectorEntry> it = Iter(a);
				while (it.hasNext()) {
					VectorEntry ve = it.next();
					int i = ve.index();
					result += ve.get() * b.get(i);
				}
			}
			else {
				Iterator<VectorEntry> it = Iter(b);
				while (it.hasNext()) {
					VectorEntry ve = it.next();
					int i = ve.index();
					result += ve.get() * a.get(i);
				}
			}
		}
		else if (a instanceof SparseVector) {
			Iterator<VectorEntry> it = Iter(a);
			while (it.hasNext()) {
				VectorEntry ve = it.next();
				int i = ve.index();
				result += ve.get() * b.get(i);
			}
		}
		else if (b instanceof SparseVector) {
			Iterator<VectorEntry> it = Iter(b);
			while (it.hasNext()) {
				VectorEntry ve = it.next();
				int i = ve.index();
				result += ve.get() * a.get(i);
			}
		}
		else result = a.dot(b);
		return result;
	}	
	
	public static double sum(Vector v) {
		double result = 0.0;
		Iterator<VectorEntry> it = Iter(v);
		while (it.hasNext()) {
			VectorEntry ve = it.next();
			result += ve.get();
		}
		return result;
	}
	
	public static double sum(double[] p) {
		double result = 0;
		for (int i = 0; i < p.length; i++) 
			result += p[i];
		return result;
	}
	
	public static Vector normalize(Vector v) {
		double sum = sum(v);
		if (sum == 0.0) uniform(v);
		else if (sum == 1.0) {}
		else scale(v, 1.0 / sum);
		return v;
	}
	
	public static void uniform(Vector v) {
		for (int i = 0; i < v.size(); i++)
			v.set(i, 1.0 / v.size());
	}
	
	public static void scale(Vector v, double x) {
		Iterator<VectorEntry> it = Iter(v);
		while (it.hasNext()) {
			VectorEntry ve = it.next();
			ve.set(ve.get() * x);
		}
	}
			
	public static boolean equal(Vector a, Vector b) {
		if (a.size() != b.size()) return false;
		for (int i = 0; i < a.size(); i++)
//			if (a.get(i) != b.get(i)) return false;
			if (Math.abs(a.get(i) - b.get(i)) > PRECISION) return false;
		return true;
	}
	
	public static boolean equal(double[] a, double[] b) {
		if (a.length != b.length) return false;
		for (int i = 0; i < a.length; i++)
			if (a[i] != b[i]) return false;
		return true;
	}
	
	public static int find(Vector target, ArrayList<Vector> list) {
		if (target == null || list.size() == 0) return -1;
		for (int i = 0; i < list.size(); i++)
			if (equal(target, list.get(i)))	return i;
		return -1;
	}
	
	public static int find(Vector target, Vector[] list) {
		if (target == null || list.length == 0) return -1;
		for (int i = 0; i < list.length; i++)
			if (equal(target, list[i])) return i;
		return -1;
	}
		
	public static int find(String target, String[] list) {
		if (target == null || list.length == 0) return -1;
		for (int i = 0; i < list.length; i++)
			if (target.equals(list[i])) return i;
		return -1;
	}

	public static int find(double target, double[] list) {
		if (list.length == 0) return -1;
		for (int i = 0; i < list.length; i++)
			if (target == list[i]) return i;
		return -1;
	}	
	
	public static int find(FscNode target, ArrayList<FscNode> set) {
		if (target == null || set.size() == 0) return FscNode.NO_INFO;
		for (int i = 0; i < set.size(); i++) 
			if (target.equals(set.get(i))) return i;
		return FscNode.NO_INFO;
	}	
	
	public static double calL1Dist(Vector a, Vector b) {
		if (a.size() != b.size()) return Double.POSITIVE_INFINITY;
		double diff = 0;
		for (int i = 0; i < a.size(); i++)
			diff += Math.abs(a.get(i) - b.get(i));
		return diff;
	}
	
	public static double calL2Dist(Vector a, Vector b) {
		if (a.size() != b.size()) return Double.POSITIVE_INFINITY;
		double diff = 0;
		for (int i = 0; i < a.size(); i++)
			diff += Math.pow(a.get(i) - b.get(i), 2);
		return Math.sqrt(diff);
	}
	
	public static void compact(Vector a) {
		if (a instanceof SparseVector) 
			((SparseVector) a).compact();
	}
	
	public static void print(ArrayList<Vector> set) {
		for (int i = 0; i < set.size(); i++) {
			Vector t = set.get(i);
			System.out.printf("%d | ", i);
			for (int j = 0; j < t.size(); j++) 
				System.out.printf("%.10f ", t.get(j));
			System.out.println();
		}
	}

	public static void print(Vector v) {
		for (int i = 0; i < v.size(); i++)
			System.out.printf("%f ", v.get(i));
		System.out.println();
	}

	
//	public static int find(Vector target, Vector[] list, int length) {
//		if (target == null || list.length == 0) return -1;
//		for (int i = 0; i < length; i++)
//			if (equal(target, list[i])) return i;
//		return -1;
//	}
	
//	public static int argmin(Vector v) {
//		int index = -1;
//		double min = Double.POSITIVE_INFINITY;
//		Iterator<VectorEntry> it = Iter(v);
//		while (it.hasNext()) {
//			VectorEntry ve = it.next();
//			if (ve.get() < min) {
//				min = ve.get();
//				index = ve.index();
//			}
//		}
//		return index;
//	}
	
//	public static int argmin(double[] p) {
//		int index = -1;
//		double min = Double.POSITIVE_INFINITY;
//		for (int i = 0 ; i < p.length; i++) {
//			if (p[i] < min) {
//				min = p[i];
//				index = i;
//			}
//		}
//		return index;
//	}
	
//	public static int argmax(Vector v) {
//		int index = -1;
//		double max = Double.NEGATIVE_INFINITY;
//		Iterator<VectorEntry> it = Iter(v);
//		while (it.hasNext()) {
//			VectorEntry ve = it.next();
//			if (ve.get() > max) {
//				max = ve.get();
//				index = ve.index();
//			}
//		}
//		return index;
//	}
	
//	public static int argmax(double[] p) {
//		int index = -1;
//		double max = Double.NEGATIVE_INFINITY;
//		for (int i = 0; i < p.length; i++) {
//			if (p[i] > max) {
//				max = p[i];
//				index = i;
//			}
//		}
//		return index;
//	}
	
//	public static double min(Vector v) {
//		double min = Double.POSITIVE_INFINITY;
//		Iterator<VectorEntry> it = Iter(v);
//		while (it.hasNext()) {
//			VectorEntry ve = it.next();
//			min = Math.min(min, ve.get());
//		}		
//		return min;
//	}
	
//	public static double min(double[] p) {
//		double min = Double.POSITIVE_INFINITY;
//		for (int i = 0; i < p.length; i++)
//			min = Math.min(min, p[i]);
//		return min;
//	}
	
//	public static double max(Vector v) {
//		double max = Double.NEGATIVE_INFINITY;
//		Iterator<VectorEntry> it = Iter(v);
//		while (it.hasNext()) {
//			VectorEntry ve = it.next();
//			max = Math.max(max, ve.get());
//		}
//		if (max == Double.NEGATIVE_INFINITY) max = 0.0;
//		return max;
//	}
	
//	public static double max(double[] p) {
//		double max = Double.NEGATIVE_INFINITY;
//		for (int i = 0; i < p.length; i++)
//			max = Math.max(max, p[i]);
//		return max;
//	}
	
//	public static double maxAbs(Vector v) {
//		double max = Double.NEGATIVE_INFINITY;
//		Iterator<VectorEntry> it = Iter(v);
//		while (it.hasNext()) {
//			VectorEntry ve = it.next();
//			max = Math.max(max, Math.abs(ve.get()));
//		}		
//		return max;
//	}
	
//	public static double maxAbs(double[] p) {
//		double max = Double.NEGATIVE_INFINITY;
//		for (int i = 0; i < p.length; i++)
//			max = Math.max(max, Math.abs(p[i]));		
//		return max;
//	}
}
