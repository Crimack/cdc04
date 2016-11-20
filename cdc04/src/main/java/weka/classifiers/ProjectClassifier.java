package weka.classifiers;

import weka.classifiers.trees.RandomForest;
import weka.core.*;

import java.util.*;

public class ProjectClassifier extends AbstractClassifier {

    private static final long serialVersionUID = 3582366333379609425L;

    // Classification variables
    private Classifier[] classifiers;
    private HashMap<Integer, Collection<Integer>> missing;

    // Instances used to track progress
    private Instances original;
    private Instances last;
    private Instances current;

    public String globalInfo() {
        return "Repeatedly applies, then rebuilds models until the output data set no longer changes.";

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        findMissingAttributes(data);
        replaceMissingValues(data);

        System.out.println(data.toString());
        original = new Instances(data);
        System.out.println("Original instantiated");
        System.out.println(original.toString());

        // Force instantiation as a quick hack to remove null errors.
        current = new Instances(data);
        last = new Instances(data);
        last.remove(0); // Forces current and last to be different on first iteration


        //trainClassifiers(data);
    }

    private void trainClassifiers(Instances data) throws Exception {
        classifiers = new Classifier[data.numAttributes()];
        for (int j = 0; j < data.numAttributes(); j++) {
            Classifier tester = new RandomForest();
            data.setClassIndex(j);
            tester.buildClassifier(data);
            classifiers[j] = tester;
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        /**
         * Should only be run through classifyDataset
         */
        Classifier active = classifiers[instance.classIndex()];
        return active.classifyInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    public Instances classifyDataset() throws Exception {
        int counter = 1;
        while (!current.toString().equals(last.toString())) {
            System.out.println("Number of iterations: " + counter);
            last = new Instances(current);
            current = new Instances(original);
            System.out.println(last.toString() == null);
            System.out.println(current.toString() == null);
            trainClassifiers(last);
            for (int i=0; i < current.numAttributes(); i++) {
                current.setClassIndex(i);
                Iterator<Integer> instanceNumbers = missing.get(i).iterator();
                while (instanceNumbers.hasNext()) {
                    int index = instanceNumbers.next();
                    Instance a = current.get(index);
                    a.setValue(i, classifyInstance(a));

                }
            }
            counter++;
            System.out.println(last.toString() == null);
            System.out.println(current.toString() == null);
            System.out.println("Lets go for another round");
            System.out.println(current.toString().equals(last.toString()));
            System.out.println();
        }
        return current;
    }

    private HashMap<Integer, Collection<Integer>> findMissingAttributes(Instances instances) {
        missing = new HashMap<Integer, Collection<Integer>>();
        // Initialise a list of instances with missing attributes for every column
        for (int i = 0; i < instances.numAttributes(); i++) {
            missing.put(i, new LinkedList<Integer>());
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

    private void replaceMissingValues(Instances instances) {
        for (int i=0; i < instances.numAttributes(); i++) {
            // Build a list of potential values to pick from
            System.out.println("Setting shit up");
            Attribute targetAtt = instances.attribute(i);
            NominalAttributeInfo info = new NominalAttributeInfo()
            targetAtt.isNominal()
            HashSet<Object> potentialValueSet = new HashSet<Object>();
            System.out.println("Starting enumeration");
            for (int j=0; j < targetAtt.; j++) {
                Attribute a = instances.get(i).;
                a.value()
                potentialValueSet.add(next);

            }
            }
            System.out.println("Finishing enumeration");
            System.out.println(potentialValueSet);
            Object[] potentialValues = potentialValueSet.toArray();
            System.out.println(potentialValues);

            // Replace the missing attribute with a random possible value
            Iterator<Integer> instanceNumbers = missing.get(i).iterator();
            while (instanceNumbers.hasNext()) {
                int index = instanceNumbers.next();
                Instance a = instances.get(index);
                Object value = potentialValues[(int) (Math.random() * potentialValues.length)];
                System.out.println("Value:" + value.toString());
                a.setValue(i, (Double) value);
            }
        }
    }


    public static void main(String[] args) {
        runClassifier(new ProjectClassifier(), args);
    }

}
