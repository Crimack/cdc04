package cdc04;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;

public class StateAnalyser implements Serializable {

	private static final long serialVersionUID = -3287661327385866977L;
	private ArrayList<Instances> tracker;

	public StateAnalyser() {
		tracker = new ArrayList<>();
	}

	private double[][] convertToMatrix(Instances a) {
		double[][] matrix = new double[a.numInstances()][a.numAttributes()];
		for (int i = 0; i < a.numInstances(); i++) {
			Instance row = a.get(i);
			matrix[i] = row.toDoubleArray();
		}
		return matrix;
	}

	public int getNumberDifferences() {
		/**
		 * Returns an integer representation of the number of rows which have
		 * changed between the previous two iterations of the classifier
		 * 
		 * A negative return values shows that the number of differences could
		 * not be calculated
		 */
		if (tracker.size() >= 2) {
			int differences = 0;
			double[][] a = convertToMatrix(tracker.get(tracker.size() - 1));
			double[][] b = convertToMatrix(tracker.get(tracker.size() - 2));
			for (int i = 0; i < a.length; i++) {
				if (!Arrays.equals(a[i], b[i]))
					differences++;
			}

			return differences;

		}

		else
			return -1;
	}
	
	public int getNumberIterations() {
		return tracker.size();
	}

	public void addInstances(Instances toAdd) {
		tracker.add(toAdd);
	}

}
