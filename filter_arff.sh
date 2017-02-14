#!/bin/bash
# Bash script to remove percentage of values from all test files
# Works if run from top level of repo and weka.jar is built

JAR_LOCATION="Code/Weka/dist/weka.jar"
INPUT_FILES=`du Test\ Files/* | cut -f2`

SAVEIFS=$IFS
IFS="$(echo -en "\n\b")"
for f in $INPUT_FILES
do
save_location="Filtered "$f
java -classpath $JAR_LOCATION weka.filters.unsupervised.attribute.ReplaceWithMissingValue \
-R first-last \
-S 1 \
-P 0.1 \
-unset-class-temporarily \
-i $f \
-o $save_location\

done
IFS=$SAVEIFS
