import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["GOTO_NUM_THREADS"] = "1"

from MI import genCasesSVMCrossEval, mainExperiments, genData, plotAll, evalMI, recombine 
from genResultsPaper import genCasesCrossEvalStatTest
from multiprocessing import Pool
import torch
import random
from EvalUtils import loadModel
from EvalUtils import getLabels, getFalseAlarmRate
from itertools import repeat
import yaml
from ExpConfigs.config_loader import _resolve_runtime_args, load_runs



torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    (file_path,) = _resolve_runtime_args()
    exp_name, runs = load_runs(file_path)
    print(f"Experiment '{exp_name}' -> {len(runs)} runs")
    for i, cfg in enumerate(runs):
        # Bind to your existing variable names
        model12      = cfg["model12"]
        randomMask   = cfg["randomMask"]
        removeNan    = cfg["removeNan"]
        graphWise    = cfg["graphWise"]
        nodeWise     = cfg["nodeWise"]
        topologyOnly = cfg["topologyOnly"]
        error_based  = cfg["error_based"]
        ones         = cfg["ones"]
        featureNames = cfg["featureNames"]
        generate     = cfg["generate"]
        ablation = cfg["ablation"]
        modelComparison = cfg["modelComparison"]
        leaveOneOut = cfg["leaveOneOut"]

        for key in cfg:
            print(key, cfg[key])

        if generate:
            mainExperiments(graphWise=graphWise,nodeWise=nodeWise, model12=model12, randomMask=randomMask, topologyOnly=topologyOnly, featureNames=featureNames, removeNan=removeNan, plot=False, MI=False, MIAblation=ablation, runIndividualFeatures=False,step='genserial',error_based=error_based,leaveOneOut=leaveOneOut, modelComparison=modelComparison, ones=ones)
        else:
            mainExperiments(graphWise=graphWise,nodeWise=nodeWise, model12=model12, randomMask=randomMask, topologyOnly=topologyOnly, featureNames=featureNames, removeNan=removeNan, plot=False, MI=True, MIAblation=ablation, runIndividualFeatures=False,step='load',error_based=error_based,leaveOneOut=leaveOneOut, modelComparison=modelComparison, ones=ones)

if __name__ == "__main__":
    run()
  
    #runExperimentsData()
