package kr.ac.kaist.pomdp.pbpi.test;

import junit.framework.TestCase;
import kr.ac.kaist.pomdp.data.GenProblems;
import kr.ac.kaist.pomdp.data.PomdpFile;
import kr.ac.kaist.pomdp.data.PomdpProblem;

public class TestGenProblems extends TestCase {
	
	public void testRun() throws Exception {
		String probName = "RockSample_4_3";
		String pomdpFileName = "./problems/" + probName + ".pomdp";
		PomdpProblem pomdpProb = PomdpFile.read(pomdpFileName, true);
		pomdpProb.printBriefInfo();
		
		GenProblems.modifyRockSample2(pomdpProb);
		
		String probName2 = "RockSample_4x3";
		String pomdpFileName2 = "./problems/" + probName2 + ".pomdp";
		PomdpProblem pomdpProb2 = PomdpFile.read(pomdpFileName2, true);
		pomdpProb2.printBriefInfo();
	}
}
