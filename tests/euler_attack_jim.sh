#!/bin/bash

# git checkout redo_bounds
bsub -n 10 -W 1:00 -J "attack" -R "rusage[mem=3072]" -e attack_err_log '--pairwise 0 --nr_eps 5 --n_jobs_per_digit 5 --maxsec 120 --check_smaller 1 --n_digits 2 --n_jobs 2'  