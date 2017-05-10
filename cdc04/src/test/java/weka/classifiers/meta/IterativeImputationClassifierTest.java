package weka.classifiers.meta;

import junit.framework.Test;
import junit.framework.TestSuite;
import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

public class IterativeImputationClassifierTest extends AbstractClassifierTest {

	public IterativeImputationClassifierTest(String name) {
		super(name);
	}

	/** Creates a default ProjectClassifier */
	public Classifier getClassifier() {
		return new IterativeImputationClassifier();
	}

	public static Test suite() {
		return new TestSuite(IterativeImputationClassifierTest.class);
	}
	
	public static void main(String[] args) {
		junit.textui.TestRunner.run(suite());
	}

}
