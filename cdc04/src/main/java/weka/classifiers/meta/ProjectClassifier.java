package weka.classifiers.meta;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Vector;
import java.util.concurrent.ThreadLocalRandom;

import cdc04.StateAnalyser;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 * A classifier which is iteratively trained, imputing missing values into
 * copies of the training data until no further change is observed. Builds one
 * learner per attribute, and therefore can take quite a while to run.
 * 
 * <!-- options-start --> Valid options are:
 * 
 * <pre>
 * -W classifier
 * Full path to the target classifier to use, e.g. weka.classifiers.trees.J48
 * </pre>
 * 
 * <pre>
 * -S
 * Defines whether or not the classifier will impute a value for the class
 * attribute as it trains.
 * </pre>
 * 
 * <pre>
 * -R
 * If set, the classifiers will be trained with any missing arguments filled in
 * by random data. The classifier will then only iterate once.
 * </pre>
 * 
 * <pre>
 * -M integer
 * Sets the maximum number of times that a particular classifier will iterate
 * before determining that it is trained.
 * </pre>
 * 
 * Options after -- are passed to the currently selected classifier.
 *
 * <!-- options-end -->
 * 
 * @author Christopher McKee (cmckee41@qub.ac.uk)
 *
 */
public class ProjectClassifier extends SingleClassifierEnhancer implements IterativeClassifier {

	/** For serialization */
	private static final long serialVersionUID = 3582366333379609425L;

	/**
	 * An array of String objects containing arguments to be passed to the
	 * classifier being used
	 */
	private String[] m_ClassifierOptions;

	/**
	 * Boolean which determines how the iterative training is performed. If
	 * false, then the class attribute is treated as though it is any other, and
	 * is iteratively imputed. If true, then the class attribute is only imputed
	 * once every other attribute has converged.
	 */
	private boolean m_Supervised = false;

	/**
	 * Boolean which simply fills missing data with random values. Useful in
	 * experiments as a control.
	 */
	private boolean m_RandomData = false;

	/**
	 * The maximum number of times which the classifier iterates over the
	 * training set. Should be set to a value lower than 50 for
	 * non-probabilistic classifiers.
	 */
	private int m_MaxIterations = Integer.MAX_VALUE;
	private int m_NumHiddenVariables;

	// Classification variables
	/** The current set of trained classifiers being iteratively trained */
	private Classifier[] m_Classifiers;
	/**
	 * A record of all of values which were originally missing from the training
	 * set. The index in the ArrayList corresponds with the attribute number,
	 * while integers within the LinkedList correspond with the instance number.
	 */
	private ArrayList<LinkedList<Integer>> m_Missing;

	// Instances used to track progress
	/**
	 * The attribute number which was originally set as the class attribute.
	 * This is recorded since the class attribute must be repeatedly changed
	 * during imputation.
	 */
	private int m_OriginalClassAttributeIndex = Integer.MIN_VALUE;

	/**
	 * A copy of the original training data
	 */
	private Instances m_Original;

	/**
	 * A copy of the training data, which classifiers are retrained against
	 */
	private Instances m_Last;

	/**
	 * Another copy of the training data, which is actively having missing
	 * values filled in
	 */
	private Instances m_Current;

	/**
	 * Object used to track the number of rows which have changed between
	 * iterations of training
	 */
	private StateAnalyser m_Tracker;

	/**
	 * Constructor
	 */
	public ProjectClassifier() {
		super();
		m_Classifier = new NaiveBayes();
	}

	/**
	 * String describing default classifier.
	 */
	@Override
	protected String defaultClassifierString() {

		return "weka.classifiers.bayes.NaiveBayes";
	}

	/**
	 * Global information about the class
	 * 
	 * @return information about the classifier which is displayed in the
	 *         CLI/GUI
	 */
	public String globalInfo() {
		return "Repeatedly applies, then rebuilds models until the output data set no longer changes.";

	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
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

		newVector.addElement(new Option(
				"\tSets the classifier to be used, and any options to be passed to it. Defaults to Naive Bayes.", "W",
				1, "-W"));

		newVector.addElement(
				new Option("\tIf set, this causes the classifier should to run as  a supervised classifier.\n"
						+ "\tIf not present, this will be supervised.", "S", 0, "-S"));

		newVector.addElement(new Option(
				"\tIf set, the classifiers will be trained with any missing arguments filled in by random data.", "R",
				0, "-R"));

		newVector.addElement(new Option("\tSets the maximum number of times that a particular classifier will iterate "
				+ "before determining that it is trained.", "M", 1, "-M"));

		newVector.addElement(new Option(
				"\tSets the number of 'hidden variables' to be inferred by the chosen classifier.", "h", 1, "-h"));

		newVector.addElement(
				new Option("", "", 0, "\nOptions specific to classifier " + m_Classifier.getClass().getName() + ":"));
		newVector.addAll(Collections.list(((OptionHandler) m_Classifier).listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options. Valid options are:
	 * <p>
	 * 
	 * -W classifier<br>
	 * Full path to the target classifier to use, e.g.
	 * weka.classifiers.trees.J48
	 * <p>
	 * 
	 * -S <br>
	 * Defines whether or not the classifier will impute a value for the class
	 * attribute as it trains.
	 * <p>
	 * 
	 * -R <br>
	 * If set, the classifiers will be trained with any missing arguments filled
	 * in by random data. The classifier will then only iterate once.
	 * <p>
	 * 
	 * -M integer <br>
	 * Sets the maximum number of times that a particular classifier will
	 * iterate before determining that it is trained.
	 * <p>
	 * 
	 * Options after -- are passed to the currently selected classifier.
	 * 
	 * @param options
	 *            The list of options as an array of Strings
	 * @exception Exception
	 *                if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		super.setOptions(options);

		setClassifierOptions(Utils.partitionOptions(options));

		setSupervised(Utils.getFlag("S", options));

		setRandomData(Utils.getFlag("R", options));

		String maxIterString = Utils.getOption("M", options);
		if (!maxIterString.isEmpty())
			setMaxIterations(Integer.parseInt(maxIterString));

		setNumHiddenVariables(Integer.parseInt(Utils.getOption('h', options)));

	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {

		ArrayList<String> options = new ArrayList<>();
		String[] superOptions = super.getOptions();

		if (m_Supervised)
			options.add("-S");

		if (m_RandomData)
			options.add("-R");

		if (m_MaxIterations < Integer.MAX_VALUE && m_MaxIterations > 0) {
			options.add("-M");
			options.add("" + m_MaxIterations);
		}
		if (m_NumHiddenVariables > 0)
			options.add("-h");
			options.add("" + m_NumHiddenVariables);

		options.addAll(Arrays.asList(superOptions));
		String[] result = new String[options.size()];

		return options.toArray(result);
	}

	/**
	 * Builds a set of classifiers based on the training data. These are
	 * iteratively trained on copies of the data.
	 * 
	 * @param data
	 *            the Instances object which comprises the training data
	 * @exception Exception
	 *                exception thrown is raised to a Weka error handler
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		m_OriginalClassAttributeIndex = instances.classIndex();
		m_Original = new Instances(instances);
		if (getNumHiddenVariables() > 0)
			inferHiddenAttributes(m_Original);
		findMissingAttributes(m_Original);
		initializeClassifier(m_Original);
		while (next()) {
		}
		done();
	}

	/**
	 * Parses a given set of Instances, usually the training data, and returns
	 * an ArrayList of LinkedLists which contains informtation about which
	 * values are missing. The ArrayList index indicates the attribute index,
	 * while the integers contained in the LinkedList objects are instance
	 * numbers.
	 * 
	 * @param instances
	 *            an Instances object, usually the training data.
	 * @return list of lists containing information about missing values
	 */
	private ArrayList<LinkedList<Integer>> findMissingAttributes(Instances instances) {
		m_Missing = new ArrayList<>();
		// Initialise a list of instances with missing attributes for every
		// column
		for (int i = 0; i < instances.numAttributes(); i++) {
			m_Missing.add(i, new LinkedList<Integer>());
		}
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance test = instances.get(i);
			for (int j = 0; j < test.numAttributes(); j++) {
				// Add the index of the instance which contains the attribute to
				// that particular attribute's list
				if (test.isMissing(j)) {
					m_Missing.get(j).add(i);
				}
			}
		}
		return m_Missing;
	}

	private void inferHiddenAttributes(Instances instances) {
		ArrayList<String> hiddenValues = new ArrayList<>();
		hiddenValues.add("1");
		hiddenValues.add("0");
		for (int i = 1; i <= getNumHiddenVariables(); i++) {
			Attribute hidden = new Attribute("hidden" + i, hiddenValues);
			instances.insertAttributeAt(hidden, m_OriginalClassAttributeIndex + i);
		}
	}

	/**
	 * Makes copies of the training data which can be mutated, and initialise
	 * the array of Classifier objects
	 * 
	 * @param instances
	 *            the training data
	 */
	public void initializeClassifier(Instances instances) throws Exception {
		// Docs say data must not be changed, so we take copies that we're free
		// to modify
		m_Tracker = new StateAnalyser();
		m_Classifiers = new Classifier[instances.numAttributes()];
		for (int i = 0; i < m_Classifiers.length; i++) {
			m_Classifiers[i] = forName(getClassifier().getClass().getName(), getClassifierOptions());
		}
		m_Current = new Instances(m_Original);
		m_Last = new Instances(m_Original);
		replaceMissingValues(m_Last);
		System.out.println(m_Last);
		m_Current.remove(0); // Force first loop to pass
	}

	/**
	 * Initially replaces missing data with random, potentially valid data by
	 * traversing m_Missing
	 * 
	 * @param instances
	 *            a dataset
	 */
	private void replaceMissingValues(Instances instances) {

		for (int i = 0; i < instances.numAttributes(); i++) {
			// Skip over the class attribute if supervised
			if (m_Supervised && i == m_OriginalClassAttributeIndex)
				continue;

			Attribute att = instances.attribute(i);
			if (att.isNominal() || att.isRelationValued() || att.isString()) {
				ArrayList<Object> a = Collections.list(att.enumerateValues());
				Iterator<Integer> instanceNumbers = m_Missing.get(i).iterator();
				while (instanceNumbers.hasNext()) {
					int index = instanceNumbers.next();
					Instance in = instances.get(index);
					String value = (String) a.get(ThreadLocalRandom.current().nextInt(a.size()));
					in.setValue(i, value);

				}
			} else {
				Iterator<Integer> instanceNumbers = m_Missing.get(i).iterator();
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

	/**
	 * Retrains each of the classifiers, then attempts to impute missing data in
	 * a copy of the training data. Does not iterate again if the results of
	 * current iteration match the results of the previous iteration, or the max
	 * number of iterations has been reached.
	 * 
	 * @return true if another iteration should be performed, otherwise false.
	 */
	public boolean next() throws Exception {
		m_Last = new Instances(m_Current);
		m_Current = new Instances(m_Original);
		retrainClassifiers(m_Last);
		if (m_RandomData)
			return false;

		for (int i = 0; i < m_Current.numAttributes(); i++) {
			if (i == m_OriginalClassAttributeIndex) {
				// Don't guess at the values of the class attribute
				continue;
			}
			m_Current.setClassIndex(i);
			Iterator<Integer> instanceNumbers = m_Missing.get(i).iterator();
			while (instanceNumbers.hasNext()) {
				int index = instanceNumbers.next();
				Instance a = m_Current.get(index);
				a.setValue(i, classifyInstance(a));

			}
		}
		m_Tracker.addInstances(m_Current);
		System.err.println(m_Tracker.getNumberIterations());
		System.err.println(m_Tracker.getNumberDifferences());
		System.err.println();

		if (m_Tracker.getNumberDifferences() == 0 || m_Tracker.getNumberIterations() >= getMaxIterations()) {
			return false;
		}
		return true;
	}

	/**
	 * Method called when iteration has terminated. Imputes class values if
	 * m_Supervised is set.
	 */
	public void done() throws Exception {
		if (m_Supervised) {
			m_Current.setClassIndex(m_OriginalClassAttributeIndex);
			for (int i : m_Missing.get(m_OriginalClassAttributeIndex)) {
				Instance toClassify = m_Current.get(i);
				toClassify.setValue(m_OriginalClassAttributeIndex, classifyInstance(toClassify));

			}
		}

		// Talk to Cassio about this. Should we remove and force a retrain or
		// just use the non-hidden classifiers
		if (getNumHiddenVariables() > 0) {
			for (int i = m_OriginalClassAttributeIndex
					+ getNumHiddenVariables(); i > m_OriginalClassAttributeIndex; i--) {
				m_Current.deleteAttributeAt(i);
			}
		}
		System.err.println(m_Current);
		System.err.println("Number of iterations taken: " + m_Tracker.getNumberIterations());
	}

	/**
	 * Retrains each classifier against a particular dataset.
	 * 
	 * @param data
	 *            training data to use
	 * @throws Exception
	 *             any exception thrown
	 */
	private void retrainClassifiers(Instances instances) throws Exception {
		for (int j = 0; j < instances.numAttributes(); j++) {
			Classifier tester = m_Classifiers[j];
			instances.setClassIndex(j);
			tester.buildClassifier(instances);
		}
	}

	/**
	 * Classifies an instance.
	 * 
	 * @param instance
	 *            the instance to classify
	 * @return the classification for the instance
	 * @throws Exception
	 *             if instance can't be classified successfully
	 */

	@Override
	public double classifyInstance(Instance instance) throws Exception {
		Classifier active = m_Classifiers[instance.classIndex()];
		return active.classifyInstance(instance);
	}

	/**
	 * Returns class probabilities for an instance.
	 * 
	 * @param instance
	 *            the instance to calculate the class probabilities for
	 * @return the class probabilities
	 * @throws Exception
	 *             if distribution can't be computed successfully
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		Classifier active = m_Classifiers[instance.classIndex()];
		return active.distributionForInstance(instance);
	}

	/**
	 * Gets classifier options
	 * 
	 * @return array of String objects to be passed to each classifier
	 */
	public String[] getClassifierOptions() {
		return m_ClassifierOptions;
	}

	/**
	 * Sets classifier options
	 * 
	 * @param classifierOptions
	 *            array of String objects to be passed to each classifier
	 */
	public void setClassifierOptions(String[] classifierOptions) {
		this.m_ClassifierOptions = classifierOptions;
	}

	/**
	 * Tip text to be displayed in the GUI for this property
	 * 
	 * @return tip text to be displayed in the GUI
	 */
	public String classifierOptionsTipText() {
		return "Options to pass to the chosen classifier";
	}

	/**
	 * Get the value of m_Supervised
	 * 
	 * @return value of m_Supervised
	 */
	public boolean getSupervised() {
		return m_Supervised;
	}

	/**
	 * Set the value of m_Supervised
	 * 
	 * @param supervised
	 *            the new value of m_Supervised
	 */
	public void setSupervised(boolean supervised) {
		this.m_Supervised = supervised;
	}

	/**
	 * Tip text to be displayed in the GUI for this property
	 * 
	 * @return tip text to be displayed in the GUI
	 */
	public String supervisedTipText() {
		return "Determines whether or not class attribute is used in training model.";
	}

	/**
	 * Get the value of m_RandomData
	 * 
	 * @return value of m_RandomData
	 */
	public boolean getRandomData() {
		return m_RandomData;
	}

	/**
	 * Set the value of m_RandomData
	 * 
	 * @param randomData
	 *            new value of m_RandomData
	 */
	public void setRandomData(boolean randomData) {
		this.m_RandomData = randomData;
	}

	/**
	 * Tip text to be displayed in the GUI for this property
	 * 
	 * @return tip text to be displayed in the GUI
	 */
	public String randomDataTipText() {
		return "Determines whether to build classifier in an iterative manner, or to just" + "use random data";
	}

	/**
	 * Get the value of m_MaxIterations
	 * 
	 * @return value of m_MaxIterations
	 */
	public int getMaxIterations() {
		return m_MaxIterations;
	}

	/**
	 * Set the value of m_MaxIterations. Defaults to Integer.MAX_VALUE if value
	 * less than 0 is supplied.
	 * 
	 * @param maxIterations
	 *            new value of m_MaxIterations
	 */
	public void setMaxIterations(int maxIterations) {
		if (maxIterations < 0)
			this.m_MaxIterations = Integer.MAX_VALUE;
		else
			this.m_MaxIterations = maxIterations;
	}

	/**
	 * Tip text to be displayed in the GUI for this property
	 * 
	 * @return tip text to be displayed in the GUI
	 */
	public String maxIterationsTipText() {
		return "Sets the maximum number of times which the classifier should iterate "
				+ "over the training data before determining that it is completed.";
	}

	public int getNumHiddenVariables() {
		return m_NumHiddenVariables;
	}

	public void setNumHiddenVariables(int numHiddenVariables) {
		this.m_NumHiddenVariables = numHiddenVariables;
	}

	public String numHiddenVariablesTipText() {
		return "Determines the number of 'hidden variables which should be inferred";
	}

	/**
	 * Main method for testing this class.
	 * 
	 * @param args
	 *            the options
	 */
	public static void main(String[] args) {
		runClassifier(new ProjectClassifier(), args);
	}

}
