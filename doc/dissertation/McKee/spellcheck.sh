#!/bin/bash

TARGET_FILES="dissertation.tex acknowledgements.tex abstract.tex intro.tex requirements.tex design_implementation.tex experiments.tex conclusion.tex datasets.tex hidden_variables.tex user_guide.tex"

for f in $TARGET_FILES
do
	aspell -t -d en_GB -c $f
done
