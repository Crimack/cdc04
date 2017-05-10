#!/bin/bash
# Bash script to remove percentage (10% by default) of values from all test files
# Accepts weka.jar path as first arg, should be run from cdc04/experiments

JAR_LOCATION=$1
INPUT_FILES=`du test_files/* | cut -f2`

SAVEIFS=$IFS
IFS="$(echo -en "\n\b")"
for f in $INPUT_FILES
do
save_location="filtered_"$f
java -classpath $JAR_LOCATION weka.filters.unsupervised.attribute.ReplaceWithMissingValue \
-R first-last \
-S 1 \
-P 0.1 \
-unset-class-temporarily \
-i $f \
-o $save_location\

done
IFS=$SAVEIFS
