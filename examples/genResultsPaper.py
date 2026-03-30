

import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
from scipy import stats
from matplotlib.lines import Line2D
from MI import recombine, getLabels
from EvalUtils import getFalseAlarmRate
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter
import pandas as pd

colors = [
    "#0E489E",  # navy #UP
    "#1E91FD",  # lighter navy #UP
    "#E6A400",  # lighter darkgoldenrod (tan-like)
    "#FFD525",  # yellow #UP
    "#00A676",  # green #UP
    "#FE9E26",   # orange  UP
    "#555555", #medium gray #UP
    "#B0B0B0" #lighter gray to completement #UP
]

temporary_storage=''
plot_directory = 'plots/MI/'

#should be
#baselines = ['GW_M12_NaNZ_nonErr_mean_FMC','GW_NaNZ_nonErr_FMCmean_FMCaUS','GW_M12_NaNZ_nonErr_var_FMCaUS','GW_NaNZ_nonErr_FMCvar_FMCaUS']
baselines = ['GW_M12_NaNZ_nonErr_FMCmean_FMCaUS','GW_NaNZ_nonErr_mean_FMC','GW_M12_NaNZ_nonErr_FMCvar_FMCaUS','GW_NaNZ_nonErr_var_FMC']
#should be
#errorBased  = ['GW_M12_NaNZ_mean_FMC','GW_NaNZ_mean_FMC','GW_M12_NaNZ_var_FMC','GW_NaNZ_var_FMC']  
# # running Exp623Large  
errorBased  = ['GW_M12_NaNZ_mean_FMC','GW_NaNZ_FMCmean_FMCaUS','GW_M12_NaNZ_var_FMC','GW_NaNZ_FMCvar_FMCaUS']
#should be
#nodeWise = ['NW_M12_NaNZ_nonErr_FMCmean_FMCaUS','NW_NaNZ_nonErr_mean_FMC']
nodeWise = ['NW_M12_NaNZ_nonErr_FMCmean_FMCaUS','NW_NaNZ_nonErr_FMCmean_FMCaUS']
#should be
#batchWise = ['M12_NaNZ_nonErr_mean_FMC','NaNZ_nonErr_mean_FMC','M12_NaNZ_nonErr_var_FMC','NaNZ_nonErr_var_FMC']
batchWise = ['M12_NaNZ_nonErr_FMCmean_FMCaUS','NaNZ_nonErr_FMCmean_FMCaUS','M12_NaNZ_nonErr_FMCvar_FMCaUS','NaNZ_nonErr_FMCvar_FMCaUS']
topoOnlies = ['GW_M12_topolOnly_NaNZ_nonErr_mean_FMC','GW_topolOnly_NaNZ_nonErr_mean_FMC','GW_M12_topolOnly_NaNZ_nonErr_var_FMC','GW_topolOnly_NaNZ_nonErr_var_FMC']
toponlOnes = ['GW_M12_topolOnlyOnes_NaNZ_nonErr_FMCmean_FMCaUS','GW_topolOnlyOnes_NaNZ_nonErr_FMCmean_FMCaUS','GW_M12_topolOnlyOnes_NaNZ_nonErr_FMCvar_FMCaUS','GW_topolOnlyOnes_NaNZ_nonErr_FMCvar_FMCaUS']
leaveOneOut = ['_Loon-1_GW_M12_NaNZ_nonErr_FMCmean','_Loon-1_GW_NaNZ_nonErr_FMCmean','_Loon-1_GW_M12_NaNZ_nonErr_FMCvar','_Loon-1_GW_NaNZ_nonErr_FMCvar']

# this shows the same results based on error, and that this works, too
def res633():
    print('############ 6.3.3 ##############')
    results = np.zeros((4,2,2))
    counter = 0
    indexes = [0,2,1,3]
    for case in baselines:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)                
                _, k = result.getKernels()
                print(k)
                acc, far = result.getBaselines()
                print(acc, far, counter, indexes[counter])
                results[indexes[counter], 0, 0] = acc
                results[indexes[counter], 1, 0] = far                
        except FileNotFoundError:
            print(temporary_storage+'interRes/All'+case+'MI.pickle not found!') 
        counter = counter+1
    counter = 0
    for case in errorBased:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)
                acc, far = result.getBaselines()
                print(acc, far)
                results[indexes[counter], 0, 1] = acc
                results[indexes[counter], 1, 1] = far  
                _, k = result.getKernels()
                print(k)
        except FileNotFoundError:
            print(temporary_storage+'interRes/All'+case+'MI.pickle not found!') 
        counter = counter+1
    #now output as latex
    models=['Small model, $\\mu$','Small model, $\\sigma$','Large model, $\\mu$','Large model, $\\sigma$']
    settings = ['Non-error based','Error-based']
    #genLtexTableFromNP(results[:,0,:], settings, models)
    print(array2latexCustom(results,main=1,col_labels=settings,row_labels=models))


# this compares graph-wise to node-wise and batch-wise comaprison
def res634():
    print('########## 6.3.4. ############')
    results = np.zeros((4,2,3))
    counter = 0
    indexes = [0,2,1,3]
    for case in baselines:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)
                acc, far = result.getBaselines()
                print(acc, far)
                results[indexes[counter], 0, 0] = acc
                results[indexes[counter], 1, 0] = far                
        except FileNotFoundError:
            print(temporary_storage+'interRes/'+case+'MI.pickle not found!') 
        counter = counter+1
    counter = 0
    for case in nodeWise:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)
                acc, far = result.getBaselines()
                print(acc, far)
                _, k = result.getKernels()
                print(k)
                results[indexes[counter], 0, 1] = acc
                results[indexes[counter], 1, 1] = far                
        except FileNotFoundError:
            print(temporary_storage+'interRes/All'+case+'MI.pickle not found!') 
        counter = counter+1
    counter = 0
    for case in batchWise:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)
                acc, far = result.getBaselines()
                print(acc, far)
                _, k = result.getKernels()
                print(k)
                results[indexes[counter], 0, 2] = acc
                results[indexes[counter], 1, 2] = far 
        except FileNotFoundError:
            print(temporary_storage+'interRes/'+case+'MI.pickle not found!') 
        counter = counter+1
    models=['Small model, $\\mu$','Small model, $\\sigma$','Large model, $\\mu$','Large model, $\\sigma$']
    settings = ['Topology','Node','Batch']
    print(results[:,0,:])
    print(results[:,1,:])
    print(array2latexCustom(results,main=1,col_labels=settings,row_labels=models))

#this shows the results when using input features =0 or =1
def sec635():
    print('########## 6.3.5. ############')
    results = np.zeros((4,2,3))
    #First the the baseline not based on error
    counter = 0
    indexes = [0,2,1,3]
    for case in baselines:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)
                acc, far = result.getBaselines()
                print(acc, far, counter, indexes[counter])
                results[indexes[counter], 0, 0] = acc
                results[indexes[counter], 1, 0] = far                
        except FileNotFoundError:
            print(temporary_storage+'interRes/'+case+'MI.pickle not found!') 
        counter = counter+1
    #features are set to 1
    counter = 0
    for case in topoOnlies:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)
                acc, far = result.getBaselines()
                print(acc, far, counter, indexes[counter])
                results[indexes[counter], 0, 1] = acc
                results[indexes[counter], 1, 1] = far   
        except FileNotFoundError:
            print(temporary_storage+'interRes/'+case+'MI.pickle not found!') 
        counter = counter+1
    #features are set to 0
    counter =0
    for case in toponlOnes:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                print(case)
                acc, far = result.getBaselines()
                print(acc, far, counter, indexes[counter])
                results[indexes[counter], 0, 2] = acc
                results[indexes[counter], 1, 2] = far   
        except FileNotFoundError:
            print(temporary_storage+'interRes/'+case+'MI.pickle not found!') 
        counter = counter+1
    models=['Small model, $\\mu$','Small model, $\\sigma$','Large model, $\\mu$','Large model, $\\sigma$']
    settings = ['Features Known','Features zeroed','Features oned']
    print(array2latexCustom(results,main=1,col_labels=settings,row_labels=models))

#this shows feasiblity and the ablation studies for GW, non error
def plots631and632():
    dataSizes = np.array([50000,25000,12500,6250,3125,1562,781,390,195,85,42,21])[::-1]
    suffixes = ['GW_M12_NaNZ_nonErr_mean','GW_NaNZ_nonErr_mean','GW_M12_NaNZ_nonErr_var','GW_NaNZ_nonErr_var']
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(4, 2.5))
    plt.gca().set_xscale('log', base=2)
    lss = ['-','--',':','-.']
    c = 0
    for suffix in suffixes:
        data = np.load(plot_directory+'files/AblationAll_'+suffix+'.npy')
        print(suffix,data[0,0],data[1,0])
        plt.plot(dataSizes, data[0][::-1], color=colors[c], label='Scores',ls=lss[0])
        plt.plot(dataSizes, data[1][::-1], color=colors[c], label='FASS',ls=lss[2])
        c = c+1
    #ax.set_xticks([20,200,2000,20000])
    plt.xticks(ticks=[10,100,1000,10000,50000], labels=['10','100','1,000','10,000','50,000'])

    handles = [
        Line2D([0], [0], color=colors[0], linestyle='-', label='Accuracy, small model, mean'),
        Line2D([0], [0], color=colors[1], linestyle='-',  label='Accuracy, large model, mean'),
        Line2D([0], [0], color=colors[2], linestyle='-', label='Accuracy, small model, variance'),
        Line2D([0], [0], color=colors[3], linestyle='-',  label='Accuracy, large Model, variance')
        #Line2D([0], [0], color=colors[0], linestyle='-', label='FAS, small model, mean'),
        #Line2D([0], [0], color=colors[1], linestyle='-',  label='FAS, large model, mean'),
        #Line2D([0], [0], color=colors[2], linestyle='-', label='FAS, small model, variance'),
        #Line2D([0], [0], color=colors[3], linestyle='-',  label='FAS, large Model, variance'),
    ]
    labels = [h.get_label() for h in handles]

    # Legend placed **inside**: middle of plot on the right
    legend = ax.legend(handles=handles, labels=labels, ncol=1,
                   loc='center right',        # right side, vertically centered
                   frameon=True, 
                   fontsize=9)

    plt.xlabel("Membership training data", fontsize=9)      # label for x-axis
    plt.ylabel("Accuracy / FAR", fontsize=9)     # label for y-ax

    plt.tight_layout()
    plt.savefig('plots/MI/A_AblationPaper.pdf')
    plt.clf()
    plt.cla()


def genAllSuffixes(emoveNan = False,randomMask= False, nodewise=True):
    params  = [[True, True, True],[True, True, False],[True, False, True],[True, False, False],
               [False, True, True],[False, True, False],[False, False, True],[False, False, False]]
    removeNan = False
    randomMask= False
    cases = []
    #check prefix
    #append mean/variance
    for case in ['mean','var']: 
        for case2 in ['', 'nonErr_']:
            for elem in params:
                suffix = 'All'
                if elem[0]:
                    suffix = suffix+'GW_'
                if elem[1]:
                    suffix = suffix+'M12_'
                if randomMask:
                    suffix = suffix+'R_'
                if elem[2]:
                    suffix = suffix+'topolOnly_'
                if not removeNan:
                    suffix = suffix+'NaNZ_'
                if nodewise and elem[0] and 'ean' in case:
                    cases.append(suffix.replace('GW','NW')+case2+case)
                cases.append(suffix+case2+case)
    return cases

def getPos(suffix):
    if 'M12' in suffix:
        if 'nonErr' in suffix:
            sec = 2
        else:
            sec = 0
    else:
        if 'nonErr' in suffix:
            sec = 3
        else:
            sec = 1
    first = 0
    if 'topolOnly' in suffix:
        first = 6
    elif 'top' in suffix:
        first = 12
    if 'GW' in suffix:
        first = first+2
    elif 'NW' in suffix:
        first = first+4
    if 'var' in suffix:
        first = first+1
    return first, sec

def genBaselinePlot():
    print('in method')
    cases = genAllSuffixes()
    #get data for plots
    performances = np.zeros((4, 18)) ##for the baselines and fas
    for case in cases:
        try: 
            with open(temporary_storage+'interRes/'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                first, second = getPos(case)
                performances[second, first], _ = result.getBaselines()
        except FileNotFoundError:
            print(temporary_storage+'interRes/'+case+'MI.pickle not found!') 
    print(performances)       
    rowNames = ['Small, error','Large, error','Small','Large']
    data_with_rows = [[rowNames[i]] + list(performances[i]) for i in range(len(performances))]
    latex_table = tabulate(data_with_rows, tablefmt="latex")
    print(latex_table)

def count_frequent_pairs(data):
    counter = Counter()
    for sublist in data:
        counter.update(sublist)
    counter.most_common()
    return counter

def res636():
    print('########## 6.3.6. ############')
    #check if indiviudal features are able to perform as well as global results
    features = ["PQ-VoltageMagnitude","PQ-VoltageAngle","PV-ReactivePowerGenerated","PV-VoltageAngle","REF-ActivePowerGenerated","REF-ReactivePowerGenerated"]
    #should be
    #suffixes = ['GW_M12_NaNZ_nonErr_mean_FMC','GW_NaNZ_nonErr_mean_FMC','GW_M12_NaNZ_nonErr_var_FMC','GW_NaNZ_nonErr_var_FMC']
    suffixes = ['GW_M12_NaNZ_nonErr_mean','GW_NaNZ_nonErr_mean','GW_M12_NaNZ_nonErr_var','GW_NaNZ_nonErr_var']
    counter = 0
    indexes = [0,2,1,3]
    smallAcc = []
    largeAcc = []
    for feature in features:
        print(feature)
        for case in suffixes:
            try:
                with open(temporary_storage+'interRes/'+feature+case+'MI.pickle', 'rb') as f:
                    result = pickle.load(f)
                    print(case)                
                    acc, far = result.getBaselines()
                    _, k = result.getKernels()
                    if 'M12' in case:
                        smallAcc.append(acc)
                    else:
                        largeAcc.append(acc)
                    if acc > 0.95:
                        print(feature, case, acc, far)
                    else:
                        print(acc, far)
                        print(k)
            except FileNotFoundError:
                print(temporary_storage+'interRes/'+feature+case+'MI.pickle not found!') 
    print(np.mean(smallAcc), np.std(smallAcc))
    print(np.mean(largeAcc), np.std(largeAcc))
           

#plots the generalization across datasets for all or for n-1 datasets
def plot637(baselineOnly=False,nmin1Only=False):
    print('############ 6.3.7 ##############')
    # original ones
    baselinesL = ['GW_M12_NaNZ_nonErr_FMCmean_FMCaUS','GW_M12_NaNZ_nonErr_FMCvar_FMCaUS','GW_NaNZ_nonErr_mean_FMC','GW_NaNZ_nonErr_var_FMC'] 
    errorBasedL  = ['GW_M12_NaNZ_mean_FMC','GW_M12_NaNZ_var_FMC','GW_NaNZ_FMCmean_FMCaUS','GW_NaNZ_FMCvar_FMCaUS']
    nodeWiseL = ['NW_M12_NaNZ_nonErr_FMCmean_FMCaUS','NW_NaNZ_nonErr_FMCmean_FMCaUS']
    batchWiseL = ['M12_NaNZ_nonErr_FMCmean_FMCaUS','M12_NaNZ_nonErr_FMCvar_FMCaUS','NaNZ_nonErr_FMCmean_FMCaUS','NaNZ_nonErr_FMCvar_FMCaUS']
    topoOnliesL = ['GW_M12_topolOnly_NaNZ_nonErr_mean_FMC','GW_M12_topolOnly_NaNZ_nonErr_var_FMC','GW_topolOnly_NaNZ_nonErr_mean_FMC','GW_topolOnly_NaNZ_nonErr_var_FMC']
    toponlOnesL = ['GW_M12_topolOnlyOnes_NaNZ_nonErr_FMCmean_FMCaUS','GW_M12_topolOnlyOnes_NaNZ_nonErr_FMCvar_FMCaUS','GW_topolOnlyOnes_NaNZ_nonErr_FMCmean_FMCaUS','GW_topolOnlyOnes_NaNZ_nonErr_FMCvar_FMCaUS']
    leaveOneOutL = ['_Loon-1_GW_M12_NaNZ_nonErr_FMCmean','_Loon-1_GW_M12_NaNZ_nonErr_FMCvar','_Loon-1_GW_NaNZ_nonErr_FMCmean','_Loon-1_GW_NaNZ_nonErr_FMCvar']
    if baselineOnly:
        suffixes = baselinesL
    elif nmin1Only:
        suffixes =leaveOneOutL
    else:
        suffixes = baselinesL +errorBasedL+nodeWiseL+batchWiseL+topoOnliesL+toponlOnesL+leaveOneOutL
    performances = np.zeros((len(suffixes),5))
    goodCombinations = []
    badCombinations = []
    pos = 0
    for case in suffixes:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                performances[pos,:]=result.Performance()
                good, neutral, bad, good_combis, bad_combis, std = result.Analysis()
                goodCombinations.append(good_combis)
                badCombinations.append(bad_combis)
        except FileNotFoundError:
                print(temporary_storage+'interRes/'+case+'MI.pickle not found!')   
        pos = pos+1
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 2))
    x = range(len(suffixes))
    print(x)
    print(np.shape(x),np.shape(performances),np.shape(performances[0]))
    plt.bar(x, performances[:,0], color=colors[0], label='Good')
    plt.bar(x, performances[:,1], bottom=performances[:,0],color=colors[1], label='Moderate')
    base = performances[:,0] + performances[:,1]
    plt.bar(x, performances[:,2], bottom=base, color=colors[4], label='Unreliable')
    base = base + performances[:,2]
    plt.bar(x, performances[:,3], bottom=base, color=colors[3], label='Poor')
    base = base + performances[:,3]
    plt.bar(x, performances[:,4], bottom=base, color=colors[2], label='Failing')

    print(performances)

    ticks = ax.get_yticks()  # e.g., [50, 75, 100, 125, 150]

    # Compute min and max for normalization
    print(ticks.min(), ticks.max(),ticks)

    ax.set_xlim(-0.5, len(suffixes)-0.5)
    # Add labels and legend
    #plt.xlabel('Bar index')
    #plt.ylabel('Value')
    #plt.title('Stacked Bar Plot from 3×n Array')
    #plt.legend()

    handles, labels = ax.get_legend_handles_labels()

    legend_handles = handles   # these already carry correct facecolor + label
    legend_labels  = labels

    ax.legend(handles=legend_handles, 
              labels=legend_labels,
              ncol=5,
              bbox_to_anchor=(0.5, 1.05),
              loc="lower center",
              fontsize=11,
              frameon=True  # draw a box
    )
    #plt.xticks(ticks=range(len(suffixes)), labels=suffixes, rotation='vertical')
    plt.yticks(ticks=[0,10,20,30,40], labels=['','20%','40%','60%','80%'])
    plt.tight_layout()
    plt.savefig('plots/MI/AA-Paper-OverallPerformance.pdf')
    plt.clf()
    plt.cla()
    print(suffixes)
    print('plotted 6.3.7')
    print('Good and bad data combinations for generalization. ########')
    print('frequent good cases.')
    print(count_frequent_pairs(goodCombinations))
    print('frequent bad cases.')
    print(count_frequent_pairs(badCombinations))
    

def genCasesCrossEvalStatTest():
    #for mean, variance, both models = everything
    cases = baselines+topoOnlies+toponlOnes+nodeWise+batchWise+leaveOneOut
    #get data for plots
    performances = np.zeros((len(cases),3)) ##for the three performances
    accuraciesUnseen = []
    accuraciesSeen = []
    falseNegRates = []
    goodCombinations = []
    badCombinations = []
    globalAccs = []
    globalFass = []
    seenNames = ['240 pserc','24 ieee','57 ieee','89 pegase','118 ieee','30']
    unseenNames = ['39 epri','60c','1354','197 snem','300','73 rts','14','5 pfm']
    unfinished = []
    for i in range(len(cases)): 
        try: 
            with open(temporary_storage+'interRes/All'+cases[i]+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
            print(cases[i])
            good, neutral, bad, good_combis, bad_combis, std = result.Analysis()
            perf, fas = result.getBaselines()
            globalAccs.append(perf)
            globalFass.append(fas)
            #print(good, neutral, bad)
            #print(np.shape(performances),np.shape(performances[i]))
            performances[i] = [good, neutral, bad]
            print(std)
            accSeen, accUnseen, farares = result.DataForTest()
            accuraciesSeen.append(accSeen)
            accuraciesUnseen.append(accUnseen)
            falseNegRates.append(farares)
            goodCombinations.append(good_combis)
            badCombinations.append(bad_combis)
            #print(case)
            #print(std)
        except FileNotFoundError:
            print(temporary_storage+'interRes/All'+cases[i]+'MI.pickle not found!')   
            #cases.remove(cases[i])  
            accuraciesSeen.append([])
            accuraciesUnseen.append([])
            falseNegRates.append([])
            goodCombinations.append([])
            badCombinations.append([])  
            unfinished.append(i)
            globalAccs.append(-0.07)
            globalFass.append(-0.07)
    #unfinished.reverse()        
    #for pos in unfinished:
    #    cases.remove(cases[pos])                         
    ###plot the amount of well performing, poorly performing, and avergae performaning generalizations 
    plt.style.use('ggplot')
    x = np.arange(len(cases))
    print(x)
    print(np.shape(x),np.shape(performances),np.shape(performances[0]))
    plt.bar(x, performances[:,0], color='green', label='Good')
    plt.bar(x, performances[:,1], bottom=performances[:,0], color='orange', label='Unreliable')
    plt.bar(x, performances[:,2], bottom=(performances[:,1]+performances[:,0]), color='r', label='Failing')
    # Add labels and legend
    #plt.xlabel('Bar index')
    #plt.ylabel('Value')
    #plt.title('Stacked Bar Plot from 3×n Array')
    #plt.legend()
    plt.xticks(ticks=range(len(cases)), labels=cases, rotation='vertical')
    plt.tight_layout()
    plt.savefig('plots/MI/A_OverallPerformanceNEW.pdf')
    plt.clf()
    plt.cla()
    plt.style.use('ggplot')
    x = np.arange(len(cases))
    print(x)
    print(np.shape(x),np.shape(performances),np.shape(performances[0]))
    plt.scatter(x, globalAccs, color='green', label='Acc')
    plt.scatter(x, globalFass, color='red', label='FAS')
    # Add labels and legend
    #plt.xlabel('Bar index')
    #plt.ylabel('Value')
    #plt.title('Stacked Bar Plot from 3×n Array')
    #plt.legend()
    plt.xticks(ticks=range(len(cases)), labels=cases, rotation='vertical')
    plt.tight_layout()
    plt.savefig('plots/MI/A_AllDatasets_OverallPerformance.pdf')
    plt.clf()
    plt.cla()
    # and run statistical test acroos all combinations
    testRes = np.ones((3,3,len(cases),len(cases)))*10.0 ##for the three performances
    for i in range(len(cases)):
        for j in range(i,len(cases)):
            if i in unfinished or j in unfinished:
                pass
            else:
                res = stats.ttest_rel(accuraciesSeen[i], accuraciesSeen[j])
                testRes[0,0,i,j] = res.statistic
                testRes[0,1,i,j] = res.pvalue
                testRes[0,2,i,j] = res.df
                res = stats.ttest_rel(accuraciesUnseen[i], accuraciesUnseen[j])
                testRes[1,0,i,j] = res.statistic
                testRes[1,1,i,j] = res.pvalue
                testRes[1,2,i,j] = res.df
                res = stats.ttest_rel(falseNegRates[i], falseNegRates[j])
                testRes[2,0,i,j] = res.statistic
                testRes[2,1,i,j] = res.pvalue
                testRes[2,2,i,j] = res.df
    print('Results based on statistical test. Significant differences. #######')

    for j in range(3):
        if j==0:
            title = 'Seen'
        elif j==1:
            title = 'Unseen'
        else:
            title = 'FAR'
        print(title)
        indx, indy = np.where(testRes[j,1]<0.05)
        for i in range(np.shape(indx)[0]):
            if not(indx[i] in unfinished or indy[i] in unfinished): 
                print(cases[indx[i]]+' and '+cases[indy[i]]+' significantly with p='+str(round(testRes[j,1,indx[i],indy[i]],3))+' and effect size '+str(round(testRes[j,0,indx[i],indy[i]],2)))
            #print(testRes[i,0])
            #print(testRes[i,1])
            #mask = testRes[i,0] > 0.05
            #print(np.where(mask, testRes[i,1], np.nan))
            ### now store results
        masked_array = np.where(testRes[j,1] == -10, np.nan, testRes)

        # Step 2: Select values where p-value < 0.05
        p_values = masked_array[j,1,:,:]  # p-values
        effect_sizes = masked_array[j,0,:,:]  #  effect sizes

        # Create a mask for significant p-values
        significant_mask = p_values < 0.05

        # Apply mask to effect sizes
        significant_effects = np.where(significant_mask, effect_sizes, np.nan)

        # Step 3: Plot heatmap of significant effect sizes
        fig = plt.imshow(significant_effects)
        
        plt.xticks(ticks=np.arange(len(cases)), labels=cases, rotation=45)
        plt.yticks(ticks=np.arange(len(cases)), labels=cases)
        plt.tight_layout()
        plt.savefig('plots/MI/A_'+title+'Stat_analysis.pdf')
        #fig.update_layout(xaxis_title="X Labels", yaxis_title="Y Labels")

    print('Good and bad data combinations for generalization. ########')
    print('frequent good cases.')
    print(count_frequent_pairs(goodCombinations))
    print('frequent bad cases.')
    print(count_frequent_pairs(badCombinations))

def res639():
    print('############ 6.3.10 ##############')
    cases = baselines+nodeWise+batchWise+toponlOnes+topoOnlies+leaveOneOut
    #for case in baselines:
    kernelsD = []
    kernels = []
    csD = []
    cs = []
    for case in cases:
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                ks, k = result.getKernels(True)
                cs, c = result.getCs(True)
                print(k, c)
                kernels.append(k)
                kernelsD = kernelsD+ks
                csD = csD+cs
                cs.append(c)
        except FileNotFoundError:
            print(temporary_storage+'interRes/All'+case+'MI.pickle not found!') 
    ### go through all cases and figure out which classifier was best, plot result per setting ()
    print(kernels)
    print(cs)
    print(kernelsD)
    print(csD)
    print('Indiviudal settings')
    print(Counter(kernels))
    print(Counter(cs))
    print('For datasets')
    print(Counter(kernelsD))
    print(Counter(csD))
    cases = baselines+leaveOneOut
    for case in cases:
        kernelsD = []
        kernels = []
        try:
            with open(temporary_storage+'interRes/All'+case+'MI.pickle', 'rb') as f:
                result = pickle.load(f)
                ks, k = result.getKernels()
                kernels.append(k)
                kernelsD=kernelsD+ks
        except FileNotFoundError:
            print(temporary_storage+'interRes/All'+case+'MI.pickle not found!') 
    ### go through all cases and figure out which classifier was best, plot result per setting ()
    print(kernels)
    print(kernelsD)
    print(Counter(kernels))
    print('For datasets')
    print(Counter(kernelsD))
    print('other settings')

def plot638(case='mean', featureList=[]):
    print('############ 6.3.9 ##############')
    cases = baselines+topoOnlies+toponlOnes+nodeWise+batchWise
    #compare: GW, BW, NW, topology, topologOnly
    results = np.ones((2,len(cases),len(cases)))*-1.0
    for i in range(len(cases)):
        seenData = []
        unseenData = []
        dataDirTest = dataDirTrain = ''
        try:
            for feature in featureList:
                if 'ea' in case:
                    dataDirTrain = temporary_storage+feature+cases[i]+'meanTrainTemp.pickle'
                else:
                    dataDirTrain = temporary_storage+feature+cases[i]+'varTrainTemp.pickle'
                with open(dataDirTrain, 'rb') as f:
                    seenData.append(pickle.load(f))
                if 'ea' in case:
                    dataDirTest = temporary_storage+feature+cases[i]+'meanEvalTemp.pickle'
                else:
                    dataDirTest = temporary_storage+feature+cases[i]+'varEvalTemp.pickle'    
                with open(dataDirTest, 'rb') as f:
                    unseenData.append(pickle.load(f))
            #sanity check Length - not neccessary, loading for feature would fail
            fulltrain = recombine(seenData,6)
            fullTest = recombine(unseenData,6)
            data = np.hstack((np.array(fulltrain),np.array(fullTest)))
            data = np.transpose(data)
            labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
            if np.shape(data)[0]>700000:
                per = 50000.0/float(np.shape(data)[0])
                _, data, _, labs = train_test_split(data, labs, test_size=per, random_state=42)
            print(np.shape(data),np.shape(labs))
            for j in range(i, len(cases)):
                try:
                    if 'ea' in case:
                        svmDir = temporary_storage+'SVM/All'+cases[j]+'mean_SVM.pickle'
                    else:
                        svmDir = temporary_storage+'SVM/All'+cases[j]+'var_SVM.pickle'
                    clf = joblib.load(svmDir)
                    res =  clf.score(data,labs)
                    fas = getFalseAlarmRate(clf,data,labs)
                    results[0,i,j]=results[0,j,i]=res
                    results[1,i,j]=results[1,j,i]=fas
                except FileNotFoundError:
                    print(svmDir+' not found!')                    
        except FileNotFoundError:
            print(dataDirTest+' or '+dataDirTrain+' not found!')
    ##add data loading??
    print(cases)
    print(results[0])
    print(results[1])
    #todo: plot this in a fancy way, maybe with Acc and FAR both in the same square, but as triangles, or something like this

def genCasesSVMCrossEval_legacy(case='mean', featureList=[]):
    params  = [[True, True, True],[True, True, False],[True, False, True],[True, False, False],
               [False, True, True],[False, True, False],[False, False, True],[False, False, False]]
    removeNan = False
    randomMask= False
    cases = []
    for elem in params:
        suffix = ''#'All_'
        if elem[0]:
            suffix = suffix+'GW_'
        if elem[1]:
            suffix = suffix+'M12_'
        if randomMask:
            suffix = suffix+'R_'
        if elem[2]:
            suffix = suffix+'topolOnly_'
        if not removeNan:
            suffix = suffix+'NaNZ_'
        cases.append(suffix)
    results = np.ones((2,len(cases),len(cases)))*-1.0
    for i in range(len(cases)):
        seenData = []
        unseenData = []
        maxIter = 6
        dataDirTest = dataDirTrain = ''
        try:
            for feature in featureList:
                if 'ea' in case:
                    dataDirTrain = temporary_storage+feature+cases[i]+'meanTrainTemp.pickle'
                else:
                    dataDirTrain = temporary_storage+feature+cases[i]+'varTrainTemp.pickle'
                with open(dataDirTrain, 'rb') as f:
                    seenData.append(pickle.load(f))
                if 'ea' in case:
                    dataDirTest = temporary_storage+feature+cases[i]+'meanEvalTemp.pickle'
                else:
                    dataDirTest = temporary_storage+feature+cases[i]+'varEvalTemp.pickle'    
                with open(dataDirTest, 'rb') as f:
                    unseenData.append(pickle.load(f))
            #sanity check Length - not neccessary, loading for feature would fail
            fulltrain = recombine(seenData,6)
            fullTest = recombine(unseenData,6)
            data = np.hstack((np.array(fulltrain),np.array(fullTest)))
            data = np.transpose(data)
            labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
            if np.shape(data)[0]>700000:
                per = 50000.0/float(np.shape(data)[0])
                _, data, _, labs = train_test_split(data, labs, test_size=per, random_state=42)
            print(np.shape(data),np.shape(labs))
            for j in range(i, len(cases)):
                try:
                    if 'ea' in case:
                        svmDir = temporary_storage+'SVM/All'+cases[j]+'mean_SVM.pickle'
                    else:
                        svmDir = temporary_storage+'SVM/All'+cases[j]+'var_SVM.pickle'
                    clf = joblib.load(svmDir)
                    res =  clf.score(data,labs)
                    fas = getFalseAlarmRate(clf,data,labs)
                    results[0,i,j]=results[0,j,i]=res
                    results[1,i,j]=results[1,j,i]=fas
                except FileNotFoundError:
                    print(svmDir+' not found!')                    
        except FileNotFoundError:
            print(dataDirTest+' or '+dataDirTrain+' not found!')
    ##add data loading??
    print(cases)
    print(results[0])
    print(results[1])


def genLtexTableFromNP(dataArray, lablesTop, lablesSide):

    df = pd.DataFrame(dataArray, index=lablesSide, columns=lablesTop)
    # Generate LaTeX table
    latex_table = df.to_latex(escape=True, index=True, float_format="%.2f")

    print(latex_table)

def array2latexCustom(
    A: np.ndarray,
    main='first',            # 'first' | 'mean' | 'sum' | int index along last axis
    float_fmt="{:.2g}",      # numeric formatting
    col_labels=None,         # list of length A.shape[1]
    row_labels=None,         # list of length A.shape[0]
    use_booktabs=True,       # \toprule/\midrule/\bottomrule vs \hline
):
    """
    Build a LaTeX table for a (R, C, K) array where each cell is:
        main_value  \textcolor{gray}{(A[i,j,1])}
    i.e., the bracket value is the *second element along the last axis*.

    Notes
    -----
    - Requires \\usepackage{xcolor} in your LaTeX preamble.
    - If K < 2, raises a ValueError (no 'second' element).
    """
    if A.ndim != 3:
        raise ValueError("A must be 3D, e.g., shape (R, C, K).")
    R, C, K = A.shape
    if K < 2:
        raise ValueError("Last axis must have at least 2 elements to use A[i,j,1].")

    if col_labels is None:
        col_labels = [f"Col {j+1}" for j in range(C)]
    if row_labels is None:
        row_labels = [f"Row {i+1}" for i in range(R)]

    def pick_main(vec):
        if isinstance(main, int):
            return vec[main]
        raise ValueError("main must be an int index")

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    align = "l" + "c"*C
    lines.append(rf"\begin{{tabular}}{{{align}}}")
    lines.append(r"\toprule" if use_booktabs else r"\hline")

    # Header
    header = " & " + " & ".join(col_labels) + r" \\"
    lines.append(header)
    lines.append(r"\midrule" if use_booktabs else r"\hline")

    print('IN GENERATING TABLE')
    # Rows
    for i in range(R):
        cells = []
        for j in range(K):
            vec = A[i, :, j]
            print(vec)
            main_val = vec[0]
            aux_val = vec[1]           # the “second” element along last axis
            main_str = float_fmt.format(float(main_val))
            aux_str  = float_fmt.format(float(aux_val))
            # gray brackets and value; requires xcolor
            cell = f"{main_str} " + r"\\farV{" + f"({aux_str})" + r"}"
            cells.append(cell)
        lines.append(row_labels[i] + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule" if use_booktabs else r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


if __name__ == "__main__": 
    #res633()
    #res634()
    #sec635()
    #res636()
    #plot637(baselineOnly=False,nmin1Only=True)
    plot637()
    #res639()
    #res6310()
    #plot638()
    #plots631and632()
    #genCasesCrossEvalStatTest()
    #resultFeasibilityOutputGran()
    #print('testtesttes')
    #genBaselinePlot()
    #genCasesCrossEvalStatTest()
    #plotAblations()
    #genCasesCrossEvalStatTest()
