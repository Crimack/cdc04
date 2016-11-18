package weka.classifiers;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class ProjectClassifier extends AbstractClassifier {

	private static final long serialVersionUID = 3582366333379609425L;
	
	private ArrayList<Classifier> classifiers;

	public String globalInfo() {
		return "Repeatedly applies, then rebuilds models until the output data set no longer changes.";
		
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		findMissingAttributes(data);
		for (int j = 0; j < data.numAttributes(); j++) {
			Classifier tester = new J48();
			data.setClassIndex(j);
			tester.buildClassifier(data);
			classifiers.set(j, tester);
		}
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
	
	private HashMap<Integer, Collection<Integer>> findMissingAttributes(Instances instances) {
		HashMap<Integer, Collection<Integer>> missing = new HashMap<Integer, Collection<Integer>>();
		// Initialise a list of instances with missing attributes for every column
		for (int i=0; i < instances.numAttributes(); i++){
			missing.put(i, new ArrayList<Integer>());
		}
		
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance test = instances.get(i);
			for (int j = 0; j < test.numAttributes(); j++) {
				// Add the index of the instance which contains the attribute to that particular
				// attribute's list
				if (test.isMissing(j)) {
					missing.get(j).add(i);
				}
			}
		}
		return missing;
	}

	public static void main (String [] args) {
		runClassifier(new ProjectClassifier(), args);
	}

}
