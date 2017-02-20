package weka.classifiers.meta;

import junit.framework.Test;
import junit.framework.TestSuite;
import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;

public class ProjectClassifierTest extends AbstractClassifierTest {

	public ProjectClassifierTest(String name) {
		super(name);
	}

	/** Creates a default ProjectClassifier */
	public Classifier getClassifier() {
		return new ProjectClassifier();
	}

	public static Test suite() {
		return new TestSuite(ProjectClassifierTest.class);
	}
	
	public static void main(String[] args) {
		junit.textui.TestRunner.run(suite());
	}

}
