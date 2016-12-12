package weka.classifiers.meta;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

import java.util.*;

public class ProjectClassifier extends AbstractClassifier implements IterativeClassifier {

	private static final long serialVersionUID = 3582366333379609425L;

	private enum ClassifierChoice {
		J48, RANDOM_FOREST, NAIVE_BAYES
	}

	private ClassifierChoice targetClassifier = ClassifierChoice.J48;

	// Classification variables
	private Classifier[] classifiers;
	private HashMap<Integer, Collection<Integer>> missing;

	// Instances used to track progress
	private int counter = 0;
	private int originalClassAttributeIndex = Integer.MIN_VALUE;
	private Instances original;
	private Instances last;
	private Instances current;

	public String globalInfo() {
		return "Repeatedly applies, then rebuilds models until the output data set no longer changes.";

	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// class
		result.disableAll();
		result.disableAllClassDependencies();
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		return result;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		originalClassAttributeIndex = data.classIndex();
		findMissingAttributes(data);
		initializeClassifier(data);
		while (next()) {
		}
		done();
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = Option.listOptionsForClassHierarchy(this.getClass(), AbstractClassifier.class);

		newVector.addElement(new Option(
				"\tIf set, specified classifier is used. Defaults to J48.\n"
						+ "\tSupported Options: J48, RandomForest, NaiveBayes. Case not important but spacing is.",
				"-C", 1, "-C"));

		return newVector.elements();
	}

	public void setOptions(String[] options) throws Exception {

		String chosenClassifier = Utils.getOption('C', options);
		if (chosenClassifier.equalsIgnoreCase("J48")) {
			setTargetClassifier(ClassifierChoice.J48);
		} else if (chosenClassifier.equalsIgnoreCase("RandomForest")) {
			setTargetClassifier(ClassifierChoice.RANDOM_FOREST);
		} else if (chosenClassifier.equalsIgnoreCase("NaiveBayes")) {
			setTargetClassifier(ClassifierChoice.NAIVE_BAYES);
		} else {
			setTargetClassifier(ClassifierChoice.J48);
		}

		super.setOptions(options);
	}

	public String[] getOptions() {

		ArrayList<String> options = new ArrayList<>();
		String[] superOptions = super.getOptions();
		options.add("-C");
		options.add("" + getTargetClassifier());

		for (String opt : superOptions) {
			options.add(opt);
		}
		String[] result = new String[options.size()];

		return options.toArray(result);
	}

	private void trainClassifiers(Instances data) throws Exception {
		System.out.println("Building new classifiers");
		classifiers = new Classifier[data.numAttributes()];
		for (int j = 0; j < data.numAttributes(); j++) {
			Classifier tester;
			switch (targetClassifier) {
			case J48:
				tester = new J48();
				break;
			case RANDOM_FOREST:
				tester = new RandomForest();
				break;
			case NAIVE_BAYES:
				tester = new NaiveBayes();
				break;
			default:
				tester = new J48();
				break;
			}
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

	public double[] distributionForInstance(Instance instance) throws Exception {
		Classifier active = classifiers[instance.classIndex()];
		return active.distributionForInstance(instance);
	}

	public void initializeClassifier(Instances instances) throws Exception {
		System.out.println("Initialising...");
		// Docs say data must not be changed, so we take copies that we're free
		// to modify
		original = new Instances(instances);
		current = new Instances(instances);
		last = new Instances(original);
		replaceMissingValues(last);
		current.remove(0); // Force first loop to pass
	}

	public boolean next() throws Exception {
		counter++;
		last = new Instances(current);
		current = new Instances(original);
		System.out.println("Number of iterations: " + counter);
		trainClassifiers(last);
		for (int i = 0; i < current.numAttributes(); i++) {
			if (i == originalClassAttributeIndex) {
				// Don't guess at the values of the class attribute
				continue;
			}
			current.setClassIndex(i);
			Iterator<Integer> instanceNumbers = missing.get(i).iterator();
			while (instanceNumbers.hasNext()) {
				int index = instanceNumbers.next();
				Instance a = current.get(index);
				a.setValue(i, classifyInstance(a));

			}
		}
		if (current.toString().equals(last.toString()) || counter >= 25) {
			return false;
		}
		System.out.println("Going again");
		System.out.println();
		return true;
	}

	public void done() throws Exception {
		System.out.println("Classifier completed");
	}

	public Instances classifyDataset() throws Exception {
		return current;
	}

	private HashMap<Integer, Collection<Integer>> findMissingAttributes(Instances instances) {
		missing = new HashMap<Integer, Collection<Integer>>();
		// Initialise a list of instances with missing attributes for every
		// column
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
		for (int i = 0; i < instances.numAttributes() - 1; i++) {
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

	public ClassifierChoice getTargetClassifier() {
		return targetClassifier;
	}

	public void setTargetClassifier(ClassifierChoice targetClassifier) {
		this.targetClassifier = targetClassifier;
	}

	public String targetClassifierTipText() {
		return "Classifier type to use. Defaults to J48";
	}

	public static void main(String[] args) {
		runClassifier(new ProjectClassifier(), args);
	}

}
