package project;

import java.util.ArrayList;

import weka.core.Instance;
import weka.core.Instances;

public class StateAnalyser {
	
	private ArrayList<Instances> tracker;
	
	public StateAnalyser() {
		tracker = new ArrayList<>();
	}
	
	private double[][] convertToMatrix(Instances a) {
		double[][] matrix = new double[a.numInstances()][a.numAttributes()];
		for (int i=0; i< a.numInstances(); i++) {
			Instance row = a.get(i);
			matrix[i] = row.toDoubleArray();
		}
		return matrix;
	}
	
	public int getNumberDifferences(){
		/**
		 * Returns an integer representation of the number of values which have changed between the previous two iterations
		 * of the classifier
		 * 
		 * A negative return values shows that the number of differences could not be calculated
		 */
		if (tracker.size() >= 2){
			int differences = 0;
			double[][] a = convertToMatrix(tracker.get(tracker.size() - 1));
			double[][] b = convertToMatrix(tracker.get(tracker.size() - 2));
			assert (a.length == b.length);
			for (int i=0; i < a[0].length; i++);)
			
		}
		
		else 
			return -1;
	}
	
	public void addInstances(Instances toAdd) {
		tracker.add(toAdd);
	}

}
