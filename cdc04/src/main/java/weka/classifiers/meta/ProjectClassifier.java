package weka.classifiers.meta;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class ProjectClassifier extends AbstractClassifier implements IterativeClassifier {

	private static final long serialVersionUID = 3582366333379609425L;

	// Classification variables
	private Classifier[] classifiers;
	private HashMap<Integer, Collection<Integer>> missing;


	// Instances used to track progress
	private int counter = 0;
	private Instances original;
	private Instances last;
	private Instances current;

	public String globalInfo() {
		return "Repeatedly applies, then rebuilds models until the output data set no longer changes.";

	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		findMissingAttributes(data);
		initializeClassifier(data);
		while (next()) {}
		done();
	}

	private void trainClassifiers(Instances data) throws Exception {
		System.out.println("Building new classifiers");
		classifiers = new Classifier[data.numAttributes()];
		for (int j = 0; j < data.numAttributes(); j++) {
			Classifier tester = new J48();
			data.setClassIndex(j);
			tester.buildClassifier(data);
			classifiers[j] = tester;
		}
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		Classifier active = classifiers[instance.classIndex()];
		return active.classifyInstance(instance);
	}

	public void initializeClassifier(Instances instances) throws Exception {
		System.out.println("Initialising...");
		// Docs say data must not be changed, so we take copies that we're free to modify
		original = new Instances(instances);
		current = new Instances(instances);
		last = new Instances(original);
		replaceMissingValues(last);
		current.remove(0); // Force first loop to pass
	}


	public boolean next() throws Exception {
		counter ++;
		last = new Instances(current);
		current = new Instances(original);
		System.out.println("Number of iterations: " + counter);
		trainClassifiers(last);
		for (int i = 0; i < current.numAttributes(); i++) {
			current.setClassIndex(i);
			Iterator<Integer> instanceNumbers = missing.get(i).iterator();
			while (instanceNumbers.hasNext()) {
				int index = instanceNumbers.next();
				Instance a = current.get(index);
				a.setValue(i, classifyInstance(a));

			}
		}
		if (current.toString().equals(last.toString())) {
			System.out.println("Classifier completed");
			return false;
		}
		System.out.println("Going again");
		System.out.println();
		return true;
	}


	public void done() throws Exception {
		System.out.println(current.toString());
	}


	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// class
		result.disableAllClasses();
		result.disableAllClassDependencies();
		result.enable(Capabilities.Capability.NOMINAL_CLASS);

		return result;
	}

	public Instances classifyDataset() throws Exception {
		return current;
	}

	private HashMap<Integer, Collection<Integer>> findMissingAttributes(Instances instances) {
		missing = new HashMap<Integer, Collection<Integer>>();
		// Initialise a list of instances with missing attributes for every column
		for (int i = 0; i < instances.numAttributes(); i++) {
			missing.put(i, new LinkedList<Integer>());
		}
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance test = instances.get(i);
			for (int j = 0; j < test.numAttributes(); j++) {
				// Add the index of the instance which contains the attribute to
				// that particular attribute's list
				if (test.isMissing(j)) {
					missing.get(j).add(i);
				}
			}
		}
		return missing;
	}

	private void replaceMissingValues(Instances instances) {
		System.out.println("Randomising missing values");
		for (int i = 0; i < instances.numAttributes(); i++) {
			// Build a list of potential values to pick from
			HashSet<Double> potentialValueSet = new HashSet<Double>();
			for (int j = 0; j < instances.numInstances(); j++) {
				Double a = instances.get(j).value(i);
				if (!a.isNaN())
					potentialValueSet.add(a);
			}
			Double[] potentialValues = potentialValueSet.toArray(new Double[potentialValueSet.size()]);

			// Replace the missing attribute with a random possible value
			Iterator<Integer> instanceNumbers = missing.get(i).iterator();
			while (instanceNumbers.hasNext()) {
				int index = instanceNumbers.next();
				Instance a = instances.get(index);
				Double value = potentialValues[(int) (Math.random() * potentialValues.length)];
				a.setValue(i, value);
			}
		}
	}

	public static void main(String[] args) {
		runClassifier(new ProjectClassifier(), args);
	}

}
