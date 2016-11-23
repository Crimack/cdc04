package weka.filters.unsupervised.attribute;

import weka.classifiers.meta.ProjectClassifier;
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
		System.out.println("Filter started");
		ProjectClassifier a = new ProjectClassifier();
		a.buildClassifier(instances);
		Instances answer = a.classifyDataset();
		System.out.println(answer.toString());
		return answer;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		return inputFormat;
	}

}
