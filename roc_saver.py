import numpy as np
import json
import os

def roc(y_pred, y_true, weights, save=False, name="", ths=None, th0=0.85, th1=1, nth=15):
    if ths==None:
        ths = np.linspace(th0, th1, nth)
    print(ths)
    x = [] # false positive raito
    y = [] # true positive ratio
    for th in ths:
       tp = np.all([y_pred>th,y_true==1], axis=0) #true positive
       tn = np.all([y_pred<th,y_true==0], axis=0) #true positive
       fp = np.all([y_pred>th,y_true==0], axis=0) #true positive
       fn = np.all([y_pred<th,y_true==1], axis=0) #true positive
       y.append(weights[tp].sum()/(weights[tp].sum()+weights[fn].sum()))
       x.append(weights[fp].sum()/(weights[fp].sum()+weights[tn].sum()))
    if save:
        if os.path.isfile("roc_curves.json"):
            with open("roc_curves.json") as file:
                curves = json.load(file)
        else:
            curves = {}
        if curves.get(name):
            print("Name already exist")
            return
        curves[name]={'x':x, 'y': y, 'th0': th0, 'th1': th1, 'nth': nth} 

        with open("roc_curves.json", "w") as file:
            json.dump(curves, file, indent=4)
    
    return (x,y)

         
