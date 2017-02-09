package weka.classifiers.meta;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

public class ProjectClassifier extends SingleClassifierEnhancer implements IterativeClassifier {

	private static final long serialVersionUID = 3582366333379609425L;
	private static final String DEFAULT_OUTPUT_DATA_PATH = "C:\\Users\\Chris\\Documents\\repos\\Project\\Test Results\\current.arff";

	private String[] classifierOptions;
	private boolean supervised = false;
	private String outputDataPath = DEFAULT_OUTPUT_DATA_PATH;

	// Classification variables
	private Classifier[] classifiers;
	private ArrayList<LinkedList<Integer>> missing;

	// Instances used to track progress
	private int counter = 0;
	private int originalClassAttributeIndex = Integer.MIN_VALUE;
	private Instances original;
	private Instances last;
	private Instances current;

	protected Classifier m_Classifier = new J48();

	/**
	 * String describing default classifier.
	 */
	@Override
	protected String defaultClassifierString() {

		return "weka.classifiers.trees.J48";
	}

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
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		return result;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = Option.listOptionsForClassHierarchy(this.getClass(), AbstractClassifier.class);

		newVector.addElement(
				new Option("\tSets the classifier to be used, and any options to be passed to it. Defaults to J48.",
						"W", 1, "-W"));

		newVector.addElement(
				new Option("\tIf set, this causes the classifier should to run as  a supervised classifier.\n"
						+ "\tIf not present, this will be supervised.", "S", 0, "-S"));

		newVector.addElement(new Option(
				"\tIf set, this will write the resulting dataset to the specified file path.\n"
						+ "\tIf not present, this will be defaulted to " + DEFAULT_OUTPUT_DATA_PATH,
				"-outputResultToFile", 1, "-outputResultToFile"));

		newVector.addElement(
				new Option("", "", 0, "\nOptions specific to classifier " + m_Classifier.getClass().getName() + ":"));
		newVector.addAll(Collections.list(((OptionHandler) m_Classifier).listOptions()));

		return newVector.elements();
	}

	public void setOptions(String[] options) throws Exception {

		super.setOptions(options);

		setClassifierOptions(Utils.partitionOptions(options));

		setSupervised(Utils.getFlag("S", options));

		String chosenWritePath = Utils.getOption("outputResultToFile", options);
		if (!chosenWritePath.isEmpty()) {
			setOutputDataPath(chosenWritePath);
		}

	}

	public String[] getOptions() {

		ArrayList<String> options = new ArrayList<>();
		String[] superOptions = super.getOptions();

		options.add("-outputResultToFile");
		options.add("" + outputDataPath);

		if (supervised)
			options.add("-S");

		options.addAll(Arrays.asList(superOptions));
		String[] result = new String[options.size()];

		return options.toArray(result);
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

	private ArrayList<LinkedList<Integer>> findMissingAttributes(Instances instances) {
		missing = new ArrayList<LinkedList<Integer>>();
		// Initialise a list of instances with missing attributes for every
		// column
		for (int i = 0; i < instances.numAttributes(); i++) {
			missing.add(i, new LinkedList<Integer>());
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
		System.out.println(missing);
		return missing;
	}

	public void initializeClassifier(Instances instances) throws Exception {
		System.out.println("Initialising...");
		// Docs say data must not be changed, so we take copies that we're free
		// to modify
		classifiers = new Classifier[instances.numAttributes()];
		for (int i = 0; i < classifiers.length; i++) {
			classifiers[i] = forName(getClassifier().getClass().getName(), getClassifierOptions());
		}
		original = new Instances(instances);
		current = new Instances(instances);
		last = new Instances(original);
		replaceMissingValues(last);
		current.remove(0); // Force first loop to pass
	}

	private void replaceMissingValues(Instances instances) {
		System.out.println("Randomising missing values");
		int attributesToReplace = instances.numAttributes();
		// Trim off the last column if supervised
		if (supervised)
			attributesToReplace--;

		for (int i = 0; i < attributesToReplace; i++) {
			// Build a list of potential values to pick from
			HashSet<Double> potentialValueSet = new HashSet<>();
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

	public boolean next() throws Exception {
		counter++;
		last = new Instances(current);
		current = new Instances(original);
		System.out.println("Number of iterations: " + counter);
		retrainClassifiers(last);
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
		if (supervised) {
			current.setClassIndex(originalClassAttributeIndex);
			for (int i : missing.get(originalClassAttributeIndex)) {
				Instance toClassify = current.get(i);
				toClassify.setValue(originalClassAttributeIndex, classifyInstance(toClassify));
			}
		}
		System.out.println("Classifier completed");
		FileWriter result = new FileWriter(outputDataPath, false);
		PrintWriter print = new PrintWriter(result);
		print.print(current.toString());
		print.close();
		result.close();
	}

	private void retrainClassifiers(Instances data) throws Exception {
		System.out.println("Building new classifiers");
		for (int j = 0; j < data.numAttributes(); j++) {
			Classifier tester = classifiers[j];
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

	public Instances classifyDataset() throws Exception {
		return current;
	}

	public String[] getClassifierOptions() {
		return classifierOptions;
	}

	public void setClassifierOptions(String[] classifierOptions) {
		this.classifierOptions = classifierOptions;
	}

	public String classifierOptionsTipText() {
		return "Options to pass to the chosen classifier";
	}

	public boolean getSupervised() {
		return supervised;
	}

	public void setSupervised(boolean supervised) {
		this.supervised = supervised;
	}

	public String supervisedTipText() {
		return "Determines whether or not class attribute is used in training model.";
	}

	public String getOutputDataPath() {
		return outputDataPath;
	}

	public void setOutputDataPath(String outputDataPath) {
		this.outputDataPath = outputDataPath;
	}

	public String outputDataPathTipText() {
		return "Determines the file location to write results to.";
	}

	public static void main(String[] args) {
		runClassifier(new ProjectClassifier(), args);
	}

}
