import os
from os.path import join
import glob
import argparse
import yaml

import models

'''
This script plots some information for all the 
model configurations inside a directory
'''

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True, help="base directory")
args = parser.parse_args()

nns = []

for root, dirs, files in os.walk(args.directory):
    for f in files:
        if f == "configs.txt":
            data = yaml.load(open(join(root, f)))
            data["version"] = root.split("/")[-1]

            #analyse model
            m = models.getModel(data["model_schema"], 34)
            nodes = []
            for l in [i for i in m.layers if not "dropout" in i.name]:
                nodes.append(int(l.output.shape[1]))
            data["nodes"] = "-".join([str(i) for i in nodes])
            data["n_weights"] = m.count_params()
            nns.append(data)

nns = sorted(nns, key=lambda d: d["AUC"], reverse=True)

for d in nns:
    print("-----------")
    print(d)

