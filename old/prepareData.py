''' 
This script saves events from a root tree to numpy array. 
The cuts and variables to extract have to be inserted in the script. 
'''
from root_numpy import tree2array
import ROOT as r
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-t', '--tree', type=str, required=True)
parser.add_argument('-n', '--nevents', type=int, required=False)
args = parser.parse_args()

f = r.TFile(args.input, "READ")
tree = f.Get(args.tree)

cuts = "(mu_eta< 2.1)&&(mu_pt > 20)&&(mjj_vjet > 65)&&(mjj_vjet < 105)&&(mjj_vbs>200)&&(deltaeta_vbs>2.5)&&(met > 20)"
#score_cuts = "(score_t0_v5>0.5)&&(score_t1>0.4)&&(score_t2>0.4)"
#cuts += "&&" + score_cuts

nn_vars = ["bveto_weights[1]", "deltaeta_mu_vbs_high",  "deltaphi_mu_vbs_high", "deltaeta_mu_vjet_high", "deltaphi_mu_vjet_high", 
              "deltaR_mu_vbs","deltaR_mu_vjet", "L_p", "L_pw", "Mww" ,"w_lep_pt", "C_ww", "C_vbs", "A_ww",
              "Zmu", "met", "mu_eta",  "mu_pt", "mjj_vjet", "deltaR_vjet", "vjet_pt_high", "A_vjet", "Rvjets_high", "Zvjets_high",
              "mjj_vbs", "deltaeta_vbs", "deltaR_vbs", "A_vbs", "vbs_pt_high",
              "N_jets_forward", "N_jets_central", "N_jets", "Ht", "R_mw", "R_ww" ]

d = tree2array(tree, selection=cuts, branches=nn_vars )
# Fixing first column
data = np.array(d.tolist(), dtype=float)

print("Data shape: ",data.shape)

if args.nevents != None: 
    np.save(args.output, data[:args.nevents])
else:
    np.save(args.output, data)

f.Close()