#!/bin/bash

#git checkout redo_bounds
bsub -n 10 -J "attack" -R "rusage[mem=3072]" -e attack_err_log   \
	'python ../attack/attacker.py --pairwise 0 --nr_eps 5 --n_jobs_per_digit 5 --maxsec 120 --check_smaller 1 --n_digits 20 --n_jobs 10'
