package weka.classifiers;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class ProjectClassifier extends AbstractClassifier {

	private static final long serialVersionUID = 3582366333379609425L;
	
	private Classifier tester = new J48();

	public String globalInfo() {
		return "Repeatedly applies, then rebuilds models until the output data set no longer changes.";
		
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	public static void main (String [] args) {
		runClassifier(new ProjectClassifier(), args);
	}

}
