
# CDC04

CDC04 is Weka plugin developed at Queen's University Belfast as part of my undergraduate final year project. The associated dissertation can also be found in the `doc` folder.

The implemented classifier iteratively trains a model by imputing missing training data, and continuing to iterate until two consecutive results are the same or the maximum iteration limit is reached. Supports all classifiers but probabilistic classifiers are recommended for best results. Also allows classifiers which do not support missing data to be used by randomly imputing missing values.

There is also some limited support for 'hidden variables,' or additional attributes which can be optionally added to the end of the test data during training.

All test files found within the experiments folder were taken from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/), and then filtered to remove around 10% of their data for testing purposes
