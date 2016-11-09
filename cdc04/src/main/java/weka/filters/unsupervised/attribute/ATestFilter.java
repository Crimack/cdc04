package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleStreamFilter;

public class ATestFilter extends SimpleStreamFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8742014203909275980L;

	@Override
	public String globalInfo() {
		return "Fires in a load of missing values for the wee thing lad";
	}

	@Override
	protected boolean hasImmediateOutputFormat() {
		return false;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		HashMap<Integer, Collection<Integer>> missing = findMissingAttributes(instances);
		System.out.println(missing);
		return instances;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected Instance process(Instance instance) throws Exception {
		System.out.println("CATS");
		// System.out.println(instance.enumerateAttributes());
		return instance;
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

}
