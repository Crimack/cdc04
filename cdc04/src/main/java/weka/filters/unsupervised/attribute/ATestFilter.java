package weka.filters.unsupervised.attribute;

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
		System.out.println(instances.numAttributes());
		System.out.println(instances.numClasses());
		System.out.println("LEMONS");
		System.out.println(instances);
		System.out.println("LIMES");
		for (Instance in : instances){
			System.out.println(in);
		}
		return instances;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected Instance process(Instance instance) throws Exception {
		System.out.println(instance.enumerateAttributes());
		return instance;
	}

}
