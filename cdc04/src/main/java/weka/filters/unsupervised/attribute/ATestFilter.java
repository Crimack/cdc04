package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

public class ATestFilter extends SimpleBatchFilter {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8742014203909275980L;

	@Override
	public String globalInfo() {
		return "Test filter which does nothing";
	}

	@Override
	protected boolean hasImmediateOutputFormat() {
		return false;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		HashMap<Integer, Collection<Integer>> missing = findMissingAttributes(instances);
		System.out.println(instances.classAttribute());
		return instances;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
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

}
