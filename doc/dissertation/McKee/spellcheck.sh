#!/bin/bash

TARGET_FILES="dissertation.tex abstract.tex intro.tex requirements.tex design_implementation.tex experiments.tex conclusion.tex datasets.tex hidden_variables.tex user_guide.tex"

for f in $TARGET_FILES
do
	aspell -c -t $f -d en
done
