package cdc04;

import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Class written to record a series of Weka instances, and to determine
 * differences between them.
 */
public class StateAnalyser implements Serializable {

	/**
	 * For serialization
	 */
	private static final long serialVersionUID = -3287661327385866977L;


	/**
	 * Most recent iteration
	 */
	private Instances current;

	/**
	 * Next most recent iteration
	 */
	private Instances last;

	/**
	 * Current total number of iterations
	 */
	private int numIterations;

	/**
	 * Constructs a new instance with no recorded instances
	 */
	public StateAnalyser() {
		numIterations = 0;
	}

	/**
	 * Takes a particular Instances object, and converts into a two-dimensional
	 * array of doubles, using the internal Weka double representation of a
	 * value. Items are accessible as matrix[x][y], where x corresponds to the
	 * instance number, and y to the attribute number.
	 * 
	 * @param instances
	 *            a set of instances to be converted
	 * @return two dimensional array of attribute values as doubles
	 */
	private double[][] convertToMatrix(Instances instances) {
		double[][] matrix = new double[instances.numInstances()][instances.numAttributes()];
		for (int i = 0; i < instances.numInstances(); i++) {
			Instance row = instances.get(i);
			matrix[i] = row.toDoubleArray();
		}
		return matrix;
	}

	/**
	 * Returns an integer representation of the number of rows which have
	 * changed between the previous two iterations of the classifier
	 *
	 * A negative return values shows that the number of differences could not
	 * be calculated, since there are not at least two iterations present.
	 */
	public int getNumberDifferences() {
		if (last != null && current != null) {
			int differences = 0;
			double[][] a = convertToMatrix(last);
			double[][] b = convertToMatrix(current);
			for (int i = 0; i < a.length; i++) {
				if (!Arrays.equals(a[i], b[i]))
					differences++;
			}

			return differences;

		}

		else
			return -1;
	}

	/**
	 * Returns the number of instances which are currently contained in the
	 * tracker for analysis
	 * 
	 * @return the current number of recorded instances
	 */
	public int getNumberIterations() {
		return numIterations;
	}

	/**
	 * Adds an Instances object to be tracked
	 * 
	 * @param toAdd
	 *            Instances object to be added
	 */
	public void addInstances(Instances toAdd) {
		last = current;
		current = toAdd;
		numIterations++;
	}

}
