package weka.classifiers.meta;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import cdc04.StateAnalyser;

public class ProjectClassifier extends SingleClassifierEnhancer implements IterativeClassifier {

	private static final long serialVersionUID = 3582366333379609425L;

	private String[] classifierOptions;
	private boolean supervised = false;
	private boolean randomData = false;
	private int maxIterations = Integer.MAX_VALUE;

	// Classification variables
	private Classifier[] classifiers;
	private ArrayList<LinkedList<Integer>> missing;

	// Instances used to track progress
	private int counter = 0;
	private int originalClassAttributeIndex = Integer.MIN_VALUE;
	private Instances original;
	private Instances last;
	private Instances current;
	private StateAnalyser tracker;

	protected Classifier m_Classifier;
	
	public ProjectClassifier() {
		super();
		tracker =  new StateAnalyser();
		m_Classifier = new J48();
	}

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
				"\tIf set, the classifiers will be trained with any missing arguments filled in by random data.", "R",
				1, "-R"));

		newVector.addElement(new Option("\tSets the maximum number of times that a particular classifier will iterate "
				+ "before determining that it is trained.", "M", 1, "-M"));

		newVector.addElement(
				new Option("", "", 0, "\nOptions specific to classifier " + m_Classifier.getClass().getName() + ":"));
		newVector.addAll(Collections.list(((OptionHandler) m_Classifier).listOptions()));

		return newVector.elements();
	}

	public void setOptions(String[] options) throws Exception {

		super.setOptions(options);

		setClassifierOptions(Utils.partitionOptions(options));

		setSupervised(Utils.getFlag("S", options));

		setRandomData(Utils.getFlag("R", options));

		String maxIterString = Utils.getOption("M", options);
		if (!maxIterString.isEmpty())
			setMaxIterations(Integer.parseInt(maxIterString));

	}

	public String[] getOptions() {

		ArrayList<String> options = new ArrayList<>();
		String[] superOptions = super.getOptions();

		if (supervised)
			options.add("-S");

		if (randomData)
			options.add("-R");

		if (maxIterations < Integer.MAX_VALUE && maxIterations > 0) {
		options.add("-M");
		options.add("" + maxIterations);}

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
		missing = new ArrayList<>();
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
		return missing;
	}

	public void initializeClassifier(Instances instances) throws Exception {
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
		int attributesToReplace = instances.numAttributes();
		// Trim off the last column if supervised
		if (supervised)
			attributesToReplace--;

		for (int i = 0; i < attributesToReplace; i++) {
			Attribute att = instances.attribute(i);
			if (att.isNominal() || att.isRelationValued() || att.isString()) {
				ArrayList<Object> a = Collections.list(att.enumerateValues());
				Iterator<Integer> instanceNumbers = missing.get(i).iterator();
				while (instanceNumbers.hasNext()) {
					int index = instanceNumbers.next();
					Instance in = instances.get(index);
					String value = (String) a.get(ThreadLocalRandom.current().nextInt(a.size()));
					in.setValue(i, value);

				}
			} else {
				Iterator<Integer> instanceNumbers = missing.get(i).iterator();
				while (instanceNumbers.hasNext()) {
					int index = instanceNumbers.next();
					Instance a = instances.get(index);
					double randomNum = ThreadLocalRandom.current().nextDouble(att.getLowerNumericBound(),
							att.getUpperNumericBound() + 1);
					a.setValue(i, randomNum);

				}
			}
		}
	}

	public boolean next() throws Exception {
		counter++;
		System.out.println(counter);
		last = new Instances(current);
		current = new Instances(original);
		retrainClassifiers(last);
		if (randomData)
			return false;

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
		tracker.addInstances(current);
		System.err.println(tracker.getNumberDifferences());
		System.out.println();

		if (current.toString().equals(last.toString()) || counter >= getMaxIterations()) {
			return false;
		}
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
		System.out.println("Number of iterations: " + counter);
	}

	private void retrainClassifiers(Instances data) throws Exception {
		for (int j = 0; j < data.numAttributes(); j++) {
			Classifier tester = classifiers[j];
			data.setClassIndex(j);
			tester.buildClassifier(data);
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

	public boolean getRandomData() {
		return randomData;
	}

	public void setRandomData(boolean randomData) {
		this.randomData = randomData;
	}

	public String randomDataTipText() {
		return "Determines whether to build classifier in an iterative manner, or to just" + "use random data";
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		if (maxIterations < 0)
			this.maxIterations = Integer.MAX_VALUE;
		else
			this.maxIterations = maxIterations;
	}

	public String maxIterationsTipText() {
		return "Sets the maximum number of times which the classifier should iterate "
				+ "over the training data before determining that it is completed.";
	}

	public static void main(String[] args) {
		runClassifier(new ProjectClassifier(), args);
	}

}
