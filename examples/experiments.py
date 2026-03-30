
from MI import genCasesSVMCrossEval, mainExperiments, genData, plotAll, evalMI, recombine 
from genResultsPaper import genCasesCrossEvalStatTest
from multiprocessing import Pool
import torch
import random
from EvalUtils import loadModel
from EvalUtils import getLabels, getFalseAlarmRate
from itertools import repeat


torch.manual_seed(0)
random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debugPlots(graphWise, model12, randomMask, topologyOnly, featureNames):
    if model12:
        model = loadModel("models/GridFM_v0_1.pth")
    else:
        model = loadModel("models/GridFM_v0_2.pth")
    #'case14_ieee'
    mean, var = genData('',model,graph_wise=graphWise, debug=True,randomMask=False,topologyOnly=topologyOnly)
    #restructure data
    for i in range(len(mean)):
        print(len(mean[i]),len(var[i]))
    if randomMask:
        evalDMean = [[] for i in repeat(None, 18)]
        evalDVar = [[] for i in repeat(None, 18)] 
    else:
        evalDMean = [[] for i in repeat(None, 6)]
        evalDVar = [[] for i in repeat(None, 6)] 
    for j in range(len(mean)):
        for i in range(3):
            evalDMean[j].append(mean[j])
            if i<2:
                evalDVar[j].append(var[j])
    for i in range(len(evalDMean)):
        print(len(evalDMean[i]),len(evalDVar[i]))
    plotAll(evalDMean,evalDVar,['t1','t2','t3'], ['v1','v2'], featureNames, 'testNode')
    #evalMI(evalDMean,evalDVar,featureNames, 'DEBUG')

def debugMIDummy(parama, paramb, paramc, paramd):
    print(parama, paramb,paramc,paramd)

def runDist(graphWise, model12,topologyOnly,nodeWise=False,gen=False):
    removeNan = False
    randomMask= False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames= featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    if gen:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,step='genserial')
    else:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,step='load')


def runExperiments(threads = 4,case=1):
    if case==1:
        params  = iter([[True, True, True],[True, True, False],[True, False, True],[True, False, False]])
    else:
        params  = iter([[False, True, True],[False, True, False],[False, False, True],[False, False, False]])
    with Pool(threads) as p:
        p.starmap(runDist, params)

def runExperimentsData():
    params  = [[True, True, True],[True, True, False],[True, False, True],[True, False, False],
               [False, True, True],[False, True, False],[False, False, True],[False, False, False]]
    for elem in params:
        runDist(elem[0],elem[1],elem[2])

def runExperimentsNodeWise(gen=False):
    graphWise = False
    nodeWise = True
    model12 = False
    randomMask= False
    topologyOnly=False
    removeNan = False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames = featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    #data_dirs_eval = [['case39_epri','case60_c'],['case1354_pegase','case197_snem'],['case300_ieee','case73_ieee_rts'],['case14_ieee','case5_pjm']]
    #data_dirs_train = [['case240_pserc'],['case24_ieee_rts', 'case57_ieee'],['case89_pegase','case118_ieee'],['case30_ieee']]      
    if gen:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='genserial')
        topologyOnly=True
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='genserial')
    else:
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load')
        topologyOnly=True
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load')
    
def runExperimentsSingleFeature():
    graphWise = True
    nodeWise = False
    model12 = False
    randomMask= False
    topologyOnly=False
    removeNan = False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames = featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load',error_based=False,modelComparison=True)
    #topologyOnly=True
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load',error_based=False,modelComparison=True)
    model12 = True
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load',error_based=False,modelComparison=True)
    #topologyOnly=False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=True,step='load',error_based=False,modelComparison=True)
    
def runGraphWise():
    runDist(True, True, True)

def runMissing():
    runDist(False, True, True)
    runDist(True, True, True)

def runSmallTopology():
    graphWise = True
    nodeWise = False
    model12 = True
    randomMask= False
    topologyOnly=True
    removeNan = False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames = featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial')
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=False)
    model12=False
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=False)

def runNodeWise(model12 = False, topologyOnly=True):
    randomMask= False
    removeNan = False
    graphWise = False
    nodeWise = True
    topologyOnly=True
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial')
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load')
    #topologyOnly=False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial')
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load')

def sec634():
    randomMask= False
    removeNan = False
    graphWise = False
    nodeWise = True
    topologyOnly=False
    model12=False
    error_based=False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True)
    model12=True
    #graphWise = False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True)
    nodeWise = False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=Ferror_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True)
    model12 = False 
    #graphWise = False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True)

def runNonErrorBased():
    randomMask= False
    removeNan = False
    graphWise = False
    nodeWise = True
    topologyOnly=False
    model12=False
    error_based=False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based)
    topologyOnly=False
    #graphWise = False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based)
    model12 = True 
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=Ferror_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based)
    topologyOnly=False
    #graphWise = False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based)

def sec657():
    #run leave 1 out without error
    leaveOneOut = True
    model12=False
    randomMask= False
    removeNan = False
    graphWise = True
    nodeWise = False
    topologyOnly=False
    error_based = False
    ones = False
    ###data is already there, only add leave -1.
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,leaveOneOut=leaveOneOut)
    model12 = True
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,leaveOneOut=leaveOneOut)
    graphWise = False
    model12 = False
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,leaveOneOut=leaveOneOut)
    model12 = True
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,leaveOneOut=leaveOneOut)



    pass

def sec633():
    model12=True
    randomMask= False
    removeNan = False
    graphWise = False
    nodeWise = False
    topologyOnly=False
    errorbased=False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if True:
        graphWise = True
        model12 = False
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
        model12 = True
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
    if False:
        #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=errorbased)
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
        model12=False
        #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=errorbased)
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
        graphWise = False
        nodeWise=True
        #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=errorbased)
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
        model12=True
        #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=Ferrorbased)
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
        nodeWise=False
        model12 = False
        #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=errorbased)
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
        model12=True
        #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=Ferrorbased)
        mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)

def sec6310():
    model12=True
    randomMask= False
    removeNan = False
    graphWise = True
    nodeWise = False
    topologyOnly=False
    errorbased=True
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True)
    # now rerun leave one out with full model comparison
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True,leaveOneOut=True)
    model12=False
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=errorbased,modelComparison=True,leaveOneOut=True)



def sec635():
    model12=False
    randomMask= False
    removeNan = False
    graphWise = True
    nodeWise = False
    topologyOnly=True
    error_based = False
    ones = False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    ##mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=error_based,ones=ones)
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,ones=ones)
    model12=True
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=error_based,ones=ones)
    ##mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,ones=ones)
    ########### Replacing with zero: above; replacing with ones: below
    ones=True
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based,ones=ones)
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True, runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,ones=ones)
    model12=False
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial',error_based=error_based,ones=ones)
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=error_based,modelComparison=True,ones=ones)


def runModelComparison(model12 = False):
    randomMask= False
    removeNan = False
    graphWise = True
    nodeWise = False
    topologyOnly=False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=False)
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=False,modelComparison=True)
    #topologyOnly=True
    #graphWise = False
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='genserial',error_based=False)
    #mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',error_based=False,modelComparison=True)

def runAllExperiments(leaveOneOut=False,modelComparison=False, ones=True,model12 = True):
    randomMask= False
    removeNan = False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames = featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    ### run once with batch wise, once with graphwise, once with node wise.
    for config in [[False, False],[True, False],[False, True]]:
        model12 = True
        topologyOnly=True
        ### first is graphwise, second is nodewise
        mainExperiments(config[0], config[1], model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial', ones=ones)
        #mainExperiments(config[0], config[1], model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',leaveOneOut=leaveOneOut,modelComparison=modelComparison, ones=ones)
        #topologyOnly=False
        model12 = False
        mainExperiments(config[0], config[1], model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,runIndividualFeatures=False,step='genserial', ones=ones)
        #mainExperiments(config[0], config[1], model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,runIndividualFeatures=False,step='load',leaveOneOut=leaveOneOut,modelComparison=modelComparison, ones=ones)

def runAblation(model12 = False):
    #take the best performing setting
    randomMask= False
    removeNan = False
    topologyOnly=False
    graphWise = False
    nodeWise = False
    featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    if randomMask:
        featureNames = featureNames+['PQ-ActivePowerDemand','PQ-ReactivePowerDemand','PQ-ActivePowerGenerated','PQ-ReactivePowerGenerated', 'PV-ActivePowerDemand','PV-ReactivePowerDemand','PV-ReactivePowerGenerated', 'PV-VoltageMagnitude','REF-ActivePowerDemand','REF-ReactivePowerDemand', 'REF-VoltageMagnitude','REF-VoltageAngle']
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,MIAblation=True, runIndividualFeatures=False,step='load',error_based=False)
    graphWise = True
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,MIAblation=True, runIndividualFeatures=False,step='load',error_based=False)
    graphWise=False
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,MIAblation=True, runIndividualFeatures=False,step='load',error_based=True)
    graphWise = True
    mainExperiments(graphWise, nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=False,MIAblation=True, runIndividualFeatures=False,step='load',error_based=True)
    graphWise=False
    mainExperiments(graphWise, nodeWise, model12, randomMask, True, featureNames, removeNan, plot=False, MI=False,MIAblation=True, runIndividualFeatures=False,step='load',error_based=False)
    graphWise = True
    mainExperiments(graphWise, nodeWise, model12, randomMask, True, featureNames, removeNan, plot=False, MI=False,MIAblation=True, runIndividualFeatures=False,step='load',error_based=False)

def debugData():
    model = loadModel("models/GridFM_v0_1.pth")
    mean, var = genData('',model,graph_wise=False, debug=True,randomMask=False,topologyOnly=False)
    unseenMean = [[] for i in repeat(None, 6)]
    print(len(mean))
    print(len(mean[0]))
    for i in range(3):
        print(i, 'iteration')
        for i in range(len(mean)):
            unseenMean[i].append(mean[i])
            #mean[i].pop(-1)
        ##this should now still be 6x something
        print(len(unseenMean))
        for i in range(len(unseenMean)):
            print(len(unseenMean[i]))
            for j in range(len(unseenMean[i])):
                print(len(unseenMean[i][j]))
    newDats = recombine(unseenMean,len(unseenMean))
    print(len(newDats))
    for i in range(len(newDats)):
            print(len(newDats[i]))
    newDats = recombine(unseenMean,len(unseenMean),2)
    print(len(newDats))
    for i in range(len(newDats)):
            print(len(newDats[i]))

def debugDataII():
    model = loadModel("models/GridFM_v0_1.pth")
    if torch.cuda.is_available():
        #move model to gpu
        model.model.to('cuda')
    print('running with eight ------------------------')
    unseenMean = [[] for i in repeat(None, 6)]
    print('before',len(unseenMean))
    data_dirs = ['case39_epri','case60_c','case1354_pegase','case197_snem','case300_ieee','case73_ieee_rts','case14_ieee','case5_pjm']
    for datadir in data_dirs:
        mean, _ = genData(datadir, model,graph_wise=False,nodeWise=False,randomMask=False,topologyOnly=False, removeNan=True)
        for i in range(len(mean)):
            unseenMean[i].append(mean[i])
        print(datadir,len(unseenMean),len(unseenMean[0]))
    print('finished, iterating')
    for i in range(len(unseenMean)):
        print(len(unseenMean[i]))
        for j in range(len(unseenMean[i])):
            print(len(unseenMean[i][j]))
    print('reproducing second case, with two only -----')
    unseenMean = [[] for i in repeat(None, 6)]
    data_dirs = ['case39_epri','case60_c']
    for datadir in data_dirs:
        mean, _ = genData(datadir, model,graph_wise=False,nodeWise=False,randomMask=False,topologyOnly=False, removeNan=True)
        for i in range(len(mean)):
            unseenMean[i].append(mean[i])
        print(datadir,len(unseenMean),len(unseenMean[0]))
    print('finished, iterating')
    for i in range(len(unseenMean)):
        print(len(unseenMean[i]))
        for j in range(len(unseenMean[i])):
            print(len(unseenMean[i][j]))



if __name__ == "__main__":
    sec635()
    #sec633()  
    #sec634()  
    #sec635()
    #sec657()
    #sec6310()
    #pass
    #featureNames = ['Active Power Demand','Reactive Power Demand','Active Power Generated','Reactive Power Generated', 'Voltage Magnitude','Voltage Angle','PQ','PV','REF']
    #runAblation()
    #genCasesCrossEvalStatTest()
    #runAblation(False)
    #runAblation(True)
    #runModelComparison()
    #runNonErrorBased()
    #runNodeWise(True, False)
    #genCasesCrossEvalStatTest()
    #runAllExperiments()
    #debugDataII()
    #runAllExperiments(False, True)
    #####LEGACY
    #debugData()
    #debugPlots(graphWise, model12, randomMask, topologyOnly, featureNames)
    #debugMI(graphWise, model12, randomMask, topologyOnly, featureNames)
    #mainExperiments(graphWise, nodeWise,model12, randomMask, topologyOnly, featureNames, removeNan, plot=True, MI=False,step='genserial')
    #mainExperiments(graphWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot=False, MI=True,step='load')
    #evalMI(['a','b','c','d','e','f','g'], ['1','2','3','4','5','6','7'],featureNames, 'Testuffix')
    #runMissing()
    #runExperiments(case=1)
    #featureNames = ['PQ-VoltageMagnitude','PQ-VoltageAngle','PV-ReactivePowerGenerated','PV-VoltageAngle','REF-ActivePowerGenerated','REF-ReactivePowerGenerated']
    #genCasesSVMCrossEval(featureList=featureNames)
    #genCasesSVMCrossEval('var',featureList=featureNames)
    #runExperimentsSingleFeature()
    #runExperimentsNodeWise()
    #runDist(False,False, False)
    #runMissing()
    #runExperimentsData()
    #runDist(False,True,True)
    #runDist(True,True,True)
    #runSmallTopology()
    #runGraphWise()
    #GW: (2417043, 6) -> Too large
    #BW:   (75541, 6) -> Fine (with test 0.2 = ~14000)