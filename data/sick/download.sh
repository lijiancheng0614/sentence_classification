#!/bin/bash
set -e

echo "Downloading SICK dataset"
mkdir -p data
cd data
wget -q -c http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip
unzip -q -o sick_train.zip
wget -c http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip
unzip -q -o sick_trial.zip
wget -c http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip
zip -q -o sick_test_annotated.zip
