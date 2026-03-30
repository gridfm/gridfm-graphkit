
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from itertools import repeat
from sklearn.metrics import confusion_matrix
import yaml

from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.training.callbacks import SaveBestModelStateDict
from gridfm_graphkit.training.plugins import MetricsTrackerPlugin
from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask
from gridfm_graphkit.io.param_handler import NestedNamespace
import torch
import random

colors = [
    "#000080",  # navy
    "#4F6D9E",  # lighter navy
    "#B8860B",  # darkgoldenrod
    "#D2B48C",  # lighter darkgoldenrod (tan-like)
    "#009E73",  # green 
    "#E69F00"   # orange 
]

def loadModel(modelPath,config_path = "config/case30_ieee_base.yaml"):
    if 'v0_1' in modelPath:
        config_path = "config/case30_ieee_baseSmall.yaml" 
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config_args = NestedNamespace(**config_dict)
    torch.manual_seed(config_args.seed)
    random.seed(config_args.seed)
    np.random.seed(config_args.seed)
    data_module = LitGridDataModule(config_args, "data")
    data_module.setup("test")
    model = FeatureReconstructionTask(config_args, data_module.node_normalizers, data_module.edge_normalizers)
    state_dict = torch.load(modelPath, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def getFalseAlarmRate(model, data, groundTruth, opposite=False):
    #part of training: label==1
    predicted=model.predict(data)
    cm=confusion_matrix(groundTruth, predicted,labels=[0,1])
    #print('computing False Alarm Rate')
    #print(cm)
    ##cm is shape 2x2, second row is test members  
    if opposite:
        #allPos = np.count_nonzero(groundTruth)
        allPos = np.sum(cm[0])
        if allPos == 0:
            far = -1.0
        else:
            far = float(cm[0,1])/float(allPos)
    else:
        #print(np.shape(groundTruth)[0],np.count_nonzero(groundTruth))
        allNegs = np.sum(cm[1])
        if allNegs ==0:
            far = -1.0
        else:
            far = float(cm[1,0])/float(allNegs)
    #print(far,np.sum(cm[1:]),float(cm[1,0])/np.sum(cm[1]))
    return np.round(far,2)

def getLabels(posnum, negnum, val=1.0):
    labels = np.zeros((posnum+negnum))
    labels[:posnum] = val
    return labels

class Result:
    def __init__(self, overallper, c, gamma, kernel, seenIndex, unseenIndex):
        self.overallper = overallper
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.otherTrain = []
        self.otherTest = []
        self.fass = []
        self.trainData = seenIndex
        self.testData = unseenIndex
    
    def addSeenResult(self, score):
        self.otherTrain.append(score)

    def addUnseenResult(self, score):
        self.otherTest.append(score)

    def addFASResult(self,score):
        self.fass.append(score)

    def getC(self):
        return self.c

    def getKernel(self):
        return self.kernel

    def getUnseen(self):
        return self.otherTest

    def getResult(self):
        return self.overallper, self.otherTrain, self.otherTest, self.fass
    
    def getSetting(self):
        return [self.c, self.gamma, self.kernel]
    
    def getSeenIndex(self):
        return self.trainData
    
    def getUnseenIndex(self):
        return self.testData
    
    def getComplete(self):
        '''
        returns in good, decent, average, bad and really bad style whether this setting reliably generalizes
        '''
        if np.min(self.otherTrain)>0.95 and np.min(self.otherTest)>0.95 and np.max(self.fass)<0.05:
            return  1.0, 0.0, 0.0, 0.0, 0.0
        if np.min(self.otherTrain)>0.90 and np.min(self.otherTest)>0.90 and np.max(self.fass)<0.1:
            return  0.0, 1.0, 0.0, 0.0, 0.0
        if np.max(self.otherTrain)<0.05 and np.max(self.otherTest)<0.05 and np.min(self.fass)>0.95:
            return 0.0, 0.0, 0.0, 0.0, 1.0
        if np.max(self.otherTrain)<0.1 and np.max(self.otherTest)<0.1 and np.min(self.fass)>0.9:
            return 0.0, 0.0, 0.0, 1.0, 0.0
        return 0.0, 0.0, 1.0, 0.0, 0.0

    def getOverall(self):
        '''
        returns in good, average, bad style whether this setting reliably generalizes
        '''
        if np.min(self.otherTrain)>0.95 and np.min(self.otherTest)>0.95 and np.max(self.fass)<0.05:
            return  1.0, 0.0, 0.0
        if np.max(self.otherTrain)<0.05 and np.max(self.otherTest)<0.05 and np.min(self.fass)>0.95:
            return 0.0, 0.0, 1.0
        return 0.0, 1.0, 0.0

    def getVariances(self):
        return np.var(self.otherTrain), np.var(self.otherTest), np.var(self.fass)


class DataStruct:
    def __init__(self, overallper, baseline, fas, c, gamma, kernel):
        self.overallper = overallper
        self.baseline = baseline
        self.fas = fas
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.otherRes = []

    def addResult(self,result):
        self.otherRes.append(result)

    def settingsStats(self):
        stats = [self.c, self.gamma, self.kernel]
        for elem in self.otherRes:
            stats = stats+elem.getSetting()
        print(Counter(stats))

    def getBaselines(self):
        return self.overallper, self.fas

    def getKernels(self,good_ones=False):
        kernels = []
        for res in self.otherRes:
            if good_ones and res.getUnseen()>0.9:
                kernels.append(res.getKernel())
            elif not good_ones:
                kernels.append(res.getKernel())
        return kernels, self.kernel

    def getCs(self,good_ones=False):
        cs = []
        for res in self.otherRes:
            if good_ones and res.getUnseen()>0.9:
                cs.append(res.getC())
            elif not good_ones:
                cs.append(res.getC())
        return cs, self.c

    def plot(self, title, suffix, directory):
        plt.style.use('ggplot')
        fig = plt.figure()
        #names for datasets
        evalMarkers = ['$39$','$60$','$1354$','$197$','$300$','$73$','$14$','$5$']
        trainMarkers = ['$240$','$24$','$57$','$89$','$118$','$30$']
        #performance overall data as line
        plt.vlines(self.overallper,0,len(self.otherRes)+2, colors=['r'],)
        plt.vlines(self.baseline,0,len(self.otherRes)+2,colors=['silver'])
        plt.vlines(self.fas,0,len(self.otherRes)+2,colors=['k'],linestyles='dashed')
        #individual settings per row, train one color, test the other
        pos = 1
        for result in self.otherRes:
            all, train, test, fas = result.getResult()
            plt.scatter(all, pos, s=40,marker='+',c='k')
            positions = [[pos] for i in repeat(None, len(train))]
            plt.scatter(train, positions, s=20,marker='s',c='Blue')
            positions = [[pos] for i in repeat(None, len(test))]
            plt.scatter(test, positions, s=17,marker='o',c='Green')
            positions = [[pos] for i in repeat(None, len(test))]
            plt.scatter(fas, positions, s=10,marker='o',c='Red')
            #now add corresponding train and test results
            plt.scatter([-0.25],[pos],s=25,marker=evalMarkers[result.getUnseenIndex()],c='k')
            plt.scatter([-0.2],[pos],s=25,marker=trainMarkers[result.getSeenIndex()],c='k')
            #pltr = plt.violinplot(train, [pos],orientation='horizontal')
            #for pc in pltr['bodies']:
            #    pc.set_facecolor('Blue')
            #plte= plt.violinplot(test, [pos],orientation='horizontal')
            #for pc in plte['bodies']:
            #    pc.set_facecolor('Green')
            pos = pos+1
        pointOr = plt.Line2D([0], [0], label='Train Performance', marker='+', markersize=7, linestyle='')
        pointTrain = plt.Line2D([0], [0], label='Seen Accuracy', marker='s', markersize=7, linestyle='', color='Blue')
        pointTest = plt.Line2D([0], [0], label='Unseen Accuracy', marker='o', markersize=7, linestyle='',color='Green')
        # add manual symbols to auto legend
        lines = [pointOr, pointTest, pointTrain]
        labels = ['Train Performance', 'Seen Accuracy', 'Unseen Accuracy']
        plt.legend(lines, labels,bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
        plt.tight_layout()
        plt.savefig(directory+title+'_'+suffix+'.pdf')
        plt.close(fig)
        plt.clf()
        plt.cla()

    def summarize(self, name, printRes=True):
         # this is the best case
        min_gen = 1.0
        alltrain = []
        alltest = []
        diffs = []
        fass = []
        for result in self.otherRes: #individual runs
            all, train, test, fas = result.getResult()
            val = np.min(train+test)
            diffs.append(all-(1.0-val))
            if val < min_gen:
                min_gen=val
            alltrain=alltrain+train
            alltest=alltest+test
            fass = fass+fas
        #get stats
        avGen = np.mean(diffs)
        medGen = np.median(diffs)
        avTrain = np.mean(alltrain)
        avTest = np.mean(alltest)
        medTrain = np.median(alltrain)
        medTest = np.median(alltest)
        minTrain = np.median(alltrain)
        minTest = np.median(alltest)

        if printRes:
            s = 'RESULT    '+name+'.  ---------------- \n'
            s = s+ 'maximum possible: '+str(round(self.overallper,3))+' with random guess'+str(round(self.baseline,3))+'\n'
            s = s+ 'worst performance: '+str(round(min_gen,3)) +'\n'+'\n'
            s = s+ 'Performance on training, mean: '+str(round(avTrain,3))+' and median: '+str(round(medTrain,3))+'\n'
            s = s+ 'Performance on test data, mean: '+str(round(avTest,3))+' and median: '+str(round(medTest,3))+'\n'
            s = s+ 'Minimal performance on train: '+str(round(minTrain,3))+' and test '+str(round(minTest,3))+'\n'
            s = s+ 'Max. generalization difference relative to base performance, mean: '+str(round(avGen,3))+' and median: '+str(round(medGen,3))
            s = s+ 'FAR on average: '+str(round(np.mean(fass),2))+', minimally '+str(round(np.min(fass),2))+', maximally '+str(round(np.max(fass),2))
            print(s)
        return self.overallper, min_gen, diffs, alltrain, alltest, fass 
    
    def DataForTest(self):
        accuraciesUnseen = []
        accuraciesSeen = []
        farares = []
        for result in self.otherRes:
            _, seen, unseen, fass = result.getResult()
            accuraciesSeen.extend(seen)
            accuraciesUnseen.extend(unseen)
            farares.extend(fass)
        return accuraciesSeen, accuraciesUnseen, farares
        
    def Performance(self):
        perfs = np.array([0.0,0.0,0.0,0.0,0.0])
        for result in self.otherRes: 
            perfs = perfs+result.getComplete()
        return perfs

    def Analysis(self):
        good_perf = 0
        av_perf = 0
        bad_perf = 0
        variances = np.array([0.0,0.0,0.0])
        good_combis = []
        bad_combis = []
        for result in self.otherRes: #individual runs 
            good, av, bad = result.getOverall()
            good_perf = good_perf+good
            av_perf = av_perf+av
            bad_perf = bad_perf+bad
            if good>0:
                good_combis.append((result.getSeenIndex(),result.getUnseenIndex()))
            elif bad>0:
                bad_combis.append((result.getSeenIndex(),result.getUnseenIndex()))
            elif av>0:
                train, test, fas = result.getVariances()
                if variances[0]==0.0: ## first entry, no averaging
                    variances[0]=train
                    variances[1]=test
                    variances[2]=fas
                else:
                    variances[0]=(variances[0]+train)/2.0
                    variances[1]=(variances[1]+test)/2.0
                    variances[2]=(variances[2]+fas)/2.0
        #we compute the square root of the variance to get the true Standard Deviation
        return good_perf, av_perf, bad_perf, good_combis, bad_combis, np.sqrt(variances)
    
            
            
