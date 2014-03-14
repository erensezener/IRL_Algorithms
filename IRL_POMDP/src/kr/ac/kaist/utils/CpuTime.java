package kr.ac.kaist.utils;

public class CpuTime {
	public static double getCurTime() {
		return (double) System.currentTimeMillis() / 1000.0;
	}
	
	public static double getElapsedTime(double startTime) {
		return (double) System.currentTimeMillis() / 1000.0 - startTime;
	}
	
}
