#!/bin/bash

for net in fc1 fc2 fc3 fc4 fc5 conv1 conv2 conv3 conv4 conv5
do
	echo Evaluating network ${net}...
	for spec in `ls ../test_cases/${net}`
	do
		timeout 10 python verifier.py --net ${net} --spec ../test_cases/${net}/${spec}
		exit_status=$?
		if [[ $exit_status -eq 124 ]]; then
			echo not verified 
		fi
	done
done


