#!/usr/bin/env python
# coding: utf-8

# In[4]:

import logging
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger()
#for hdlr in log.handlers[:]:  # remove all old handlers
#    log.removeHandler(hdlr)
fileh = logging.FileHandler('/storage/vbsjjlnu/log.txt', 'a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileh.setFormatter(formatter)
log.addHandler(fileh)

import model_opt
import argparse
import time

import GPyOpt

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n-iter', type=int, required=True, default=10)
args = parser.parse_args()

logging.info("=================================== Starting new training")

# In[83]:


config = {
    "base_dir":        "/storage/vbsjjlnu",
    "plot_config":     "Full2018v6s5",
    "cut":             "boost",
    "samples_version": "v10",
    "cols": ['mjj_vbs','deltaeta_vbs','Centr_ww','Asym_vbs','A_ww',
             'Lepton_eta','mjj_vjet','vbs_0_pt','vbs_1_pt','vjet_0_pt',
             'Lepton_pt','Centr_vbs','w_lep_pt','Mtw_lep','deltaphi_vbs',
             'Mww','PuppiMET','Lepton_flavour','R_mw','vjet_0_eta','Zlep',
             'Zvjets_0','R_ww',
        ],
    "verbose": 0
}
    
bounds = [{'name': 'batch_size', 'type': 'discrete',  'domain': (1024, 2048, 4096)},
          {'name': 'n_layers', 'type': 'discrete',  'domain': (2,3,4,5,6)},
          {'name': 'n_nodes', 'type': 'discrete',  'domain': (10,30,50,80,100,150,200)},
          {'name': 'dropout', 'type': 'discrete',  'domain': (0,0.05,0.1,0.2,0.3)},
          {'name': 'batch_norm', 'type': 'discrete',  'domain': (0,1)},
          {'name': 'input_dim', 'type': 'discrete',  'domain': tuple(list(range(4,len(config["cols"])+1) ))},
         ]


fixed_params={
    "epochs": 200,
    "val_ratio": 0.25,
    "test_ratio": 0.1,
    "patience": (0.0001, 20)
}


## optimizer function
def f(x):
    ev = model_opt.evaluate_vbsdnn_model(config, fixed_params, x)
    return ev

logging.info("Initialization")
optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

logging.info(f"Running {args.n_iter} optimization")
optimizer.run_optimization(max_iter=args.n_iter)

timeid = int(time.time())

with open(f"{config['base_dir']}/{config['plot_config']}/{config['cut']}/best_params_{timeid}.txt", "a") as of:
    for ib, b in enumerate(bounds):
        of.write(f"{b['name']} : {optimizer.x_opt[ib]}\n")
    of.write(f"Best significance: {optimizer.fx_opt}")


optimizer.save_evaluations(f"{config['base_dir']}/{config['plot_config']}/{config['cut']}/baysian_model_{timeid}")





