package cdc04;

import java.util.concurrent.ThreadLocalRandom;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class StateAnalyserTest extends TestCase {

	private StateAnalyser construct;
	private Instances a;
	private Instances b;
	private Instances c;

	protected void setUp() throws Exception {
		super.setUp();
		construct = new StateAnalyser();
		a = DataSource.read(ClassLoader.getSystemResourceAsStream("cdc04/data/audiology.arff"));
		int max = a.numAttributes();
		b = new Instances(a);
		for (Instance in : b) {
			boolean replaced = false;
			while (!replaced) {
				int attIndex = ThreadLocalRandom.current().nextInt(0, max);
				if (!in.isMissing(attIndex)) {
					in.setMissing(attIndex);
					replaced = true;
				}
			}
		}
		c = new Instances(b);
	}

	public void testStateAnalyser() {
		assertEquals(construct.getNumberDifferences(), -1);
		assertEquals(construct.getNumberIterations(), 0);

	}

	public void testGetNumberDifferences() {
		assertEquals(construct.getNumberDifferences(), -1);
		construct.addInstances(a);
		assertEquals(construct.getNumberDifferences(), -1);
		construct.addInstances(b);
		assertEquals(construct.getNumberDifferences(), a.numInstances());
		construct.addInstances(c);
		assertEquals(construct.getNumberDifferences(), 0);
	}

	public void testGetNumberIterations() {
		assertEquals(construct.getNumberIterations(), 0);
		construct.addInstances(a);
		assertEquals(construct.getNumberIterations(), 1);
		construct.addInstances(b);
		assertEquals(construct.getNumberIterations(), 2);
		construct.addInstances(c);
		assertEquals(construct.getNumberIterations(), 3);
	}

	public static Test suite() {
		return new TestSuite(StateAnalyserTest.class);
	}

	public static void main(String[] args) {
		junit.textui.TestRunner.run(suite());
	}

}
