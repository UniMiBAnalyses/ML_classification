#!/usr/bin/env python
# coding: utf-8

'''
# @ unimib

cd /storage/ML_classification

# fix config inside the script with base path

python3 dnn_bayesian_analyze.py -t 1582450453
'''

import os
import yaml
import argparse
import pprint
import numpy as np

config = {
    "base_dir":    "/storage/vbsjjlnu", # unimib
    "plot_config": "Full2018v6s5",
    "cut":         "resolved",          # unimib
}

def analyze_steps():
    config["time_start"] =  args.time_start_dir

    opt_report_dir = os.path.join( 
        config['base_dir'], 
        config['plot_config'], 
        config['cut'], 
        "bayesian_optimizations" ,
        str(config["time_start"]),
    )

    opt_metadata = yaml.load( open(os.path.join(opt_report_dir, "optimization_metadata.yml"), "r") )

    # # optimization metadata
    # pprint.pprint(opt_metadata)

    steps_dir = os.path.join(
        config['base_dir'], 
        config['plot_config'], 
        config['cut'], 
        "steps"
    )
    steps = os.listdir( steps_dir )
    steps = [step for step in steps if int(step) < int(opt_metadata["end_time"]) and int(step) > int(opt_metadata["start_time"]) ]
    steps = sorted(steps)
    # # list of the steps during this training
    # print( steps )

    steps_metadata = []
    for step in steps:
        steps_metadata.append( yaml.load( open(os.path.join(steps_dir, step, "model_metadata.yml"), "r")) )
    # # metadata of a single step
    # pprint.pprint(steps_metadata[0])

    return steps_metadata

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time-start-dir', type=int,)
    args = parser.parse_args()

    opt_reports_dir = os.path.join( 
        config['base_dir'], 
        config['plot_config'], 
        config['cut'], 
        "bayesian_optimizations" ,
    )

    if not args.time_start_dir:
        pprint.pprint( sorted(os.listdir(opt_reports_dir)) )

    if args.time_start_dir:
        steps_metadata = analyze_steps()
        ## implementare qualche nuova funzione qui
        ## fare una funzione per ogni step dell'analisi. niente lunghi script infiniti che non finiscono più


