#!/usr/bin/env python
# coding: utf-8

import logging
from telegramBot import TelegramLog

import model_opt

import argparse
import time
import os

import GPyOpt

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n-iter', type=int, required=True, default=10)
parser.add_argument('-b', '--bot-config', type=str, required=True)
args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger()
#for hdlr in log.handlers[:]:  # remove all old handlers
#    log.removeHandler(hdlr)
fileh = logging.FileHandler('/storage/vbsjjlnu/log.txt', 'a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileh.setFormatter(formatter)
log.addHandler(fileh)

bot = TelegramLog(args.bot_config)
log.addHandler(bot)


logging.info("=================================== Starting new training")

config = {
    "base_dir":        "/storage/vbsjjlnu",
    "plot_config":     "Full2018v6s5",
    "cut":             "resolved",
    "samples_version": "v10",
    "cols": ['mjj_vbs','deltaeta_vbs','Centr_ww','Asym_vbs','A_ww',
             'Lepton_eta','mjj_vjet','vbs_0_pt','vbs_1_pt','vjet_0_pt','vjet_1_pt',
             'Lepton_pt','Centr_vbs','w_lep_pt','Mtw_lep','deltaphi_vbs',
             'Mww','PuppiMET','Lepton_flavour','R_mw','vjet_0_eta','vjet_1_eta','Zlep',
             'Zvjets_0','R_ww',
        ],
    "verbose": 0
}

bounds = [{'name': 'batch_size', 'type': 'discrete',  'domain': (2048, 4096)},
          {'name': 'n_layers', 'type': 'discrete',  'domain': (2,3,4,5,6)},
          {'name': 'n_nodes', 'type': 'discrete',  'domain': (10,15,20,25,30,40,50,100,150)},
          {'name': 'dropout', 'type': 'discrete',  'domain': (0,0.05,0.1,0.2,0.3)},
          {'name': 'batch_norm', 'type': 'discrete',  'domain': (0,1)},
          {'name': 'input_dim', 'type': 'discrete',  'domain': tuple(list(range(4,len(config["cols"])+1) ))},
         ]

fixed_params={
    "epochs": 200,
    "val_ratio": 0.5,
    "test_ratio": 0.1,
    "patience": (0.0001, 20)
}

## optimizer function
def f(x):
    ev, vbs_dnn = model_opt.evaluate_vbsdnn_model(config, fixed_params, x)
    # Send image
    bot.send_image(vbs_dnn._VbsDnn__model_dir+"/model_train.png")
    return ev

logging.info("Initialization")
# try this acquisition type
# https://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_scikitlearn.ipynb
optimizer = GPyOpt.methods.BayesianOptimization(f=f, 
                                                domain=bounds,
                                                acquisition_type ='EI',       # MPI acquisition
                                                acquisition_weight = 0.2,   # Exploration exploitation
                                                jitter=0.1,
                                                )

logging.info(f"Running {args.n_iter} optimization")
optimizer.run_optimization(max_iter=args.n_iter)
logging.info("=================================== Optimization ended! ")

timeid = int(time.time())

## Save metadata about optimization

report_dir = os.path.join( config['base_dir'], config['plot_config'], config['cut'] )

with open(os.path.join(report_dir, f"{timeid}_best_params.txt"), "a") as of:
    for ib, b in enumerate(bounds):
        best_value_str = f"{b['name']} : {optimizer.x_opt[ib]}\n"
        logging.info(best_value_str)
        of.write(best_value_str)
    best_significance_str = f"Best significance: {optimizer.fx_opt}"
    logging.info(best_significance_str)
    of.write(best_significance_str)

optimizer.save_evaluations(os.path.join(report_dir, f"{timeid}_baysian_model_saveopt.txt"))
optimizer.save_report(os.path.join(report_dir, f"{timeid}_baysian_model_report.txt"))
# optimizer.plot_acquisition(filename=os.path.join(report_dir, f"{timeid}_baysian_model_acquisition.png"))
optimizer.plot_convergence(filename=os.path.join(report_dir, f"{timeid}_baysian_model_convergence.png"))
# bot.send_image(os.path.join(report_dir, f"{timeid}_baysian_model_acquisition.png"))
bot.send_image(os.path.join(report_dir, f"{timeid}_baysian_model_convergence.png"))




