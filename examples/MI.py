#import sys
#sys.path.append('~/graphs/GraphNNDabble/gridfm-graphkit/gridfm_graphkit')

from gridfm_graphkit.datasets.powergrid_dataset import GridDatasetDisk
from gridfm_graphkit.datasets.normalizers import BaseMVANormalizer
#from gridfm_graphkit.training.trainer import Trainer
from gridfm_graphkit.datasets.utils import split_dataset
from gridfm_graphkit.datasets.transforms import AddPFMask, AddRandomMask
from gridfm_graphkit.tasks.feature_reconstruction_task import FeatureReconstructionTask
from gridfm_graphkit.io.param_handler import NestedNamespace
#from gridfm_graphkit.utils.loss import PBELoss
from gridfm_graphkit.datasets.globals import PQ, PV, REF


# Standard open-source libraries
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from itertools import repeat
from EvalUtils import DataStruct, Result, getFalseAlarmRate, getLabels, loadModel
import joblib
import pickle
import copy
import yaml
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

from EvalUtils import colors

temporary_storage=''
data_directory=""
plot_directory = 'plots/MI/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plotAll(seenData,unseenData,seenNames, evalNames, titles, suffix):
    #plot(seenData[3],unseenData[3],seenNames,evalNames,titles[3],suffix)
    #plot(seenData[2],unseenData[2],seenNames,evalNames,titles[2],suffix)
    #plot(seenData[0],unseenData[0],seenNames,evalNames,titles[0],suffix)
    #plot(seenData[1],unseenData[1],seenNames,evalNames,titles[1],suffix)
    for i in range(len(titles)):
        _ = plot(seenData[i],unseenData[i],seenNames,evalNames,titles[i],suffix)

def plot(seenData,unseenData,seenNames, evalNames, title, suffix):
    #print(seenData, unseenData)
    plt.style.use('ggplot')
    fig = plt.figure()
    if len(seenData)>0:
        positionsTrain = range(len(seenNames))
        for pos in range(len(positionsTrain)):
            if len(seenData[pos])>1:
                #print([positionsTrain[pos]])
                plt.violinplot(seenData[pos],[positionsTrain[pos]])
    if len(unseenData)>0:
        positionsEval = range(len(seenNames)+2,len(seenNames)+2+len(evalNames))
        #print(positionsEval, len(unseenData))
        allLabels = seenNames+['']+evalNames
        #print(pos, allLabels)
        for pos in range(len(positionsEval)):
            if len(unseenData[pos])>1:
                #print([positionsEval[pos]])
                plt.violinplot(unseenData[pos],[positionsEval[pos]])
    pos = list(positionsTrain)+[len(seenNames)+1]+list(positionsEval)
    plt.xticks(pos, allLabels, rotation='vertical')
    plt.tight_layout()
    plt.savefig('plots/'+title+'_'+suffix+'.pdf')
    plt.close(fig)
    plt.clf()
    plt.cla()
    return 0

def addNanCorrect(tarlist, elem, removeNan):
    #print(elem,np.isnan(elem))
    if np.isnan(elem):
        if removeNan:
            pass
        else:
            tarlist.append(0.0)
    else:
        tarlist.append(elem)  
        #print('added', len(tarlist), tarlist[-1])      

def genData(dir, model,graph_wise=False,nodeWise=False,randomMask=False,removeNan=True,error_based=True,topologyOnly=False,debug=False,ones=False):
    #generates or load the data, e.g., model inputs and outputs, that can then be used to asssess membership inference risks
    data_dir = data_directory+dir
    if debug:
        data_dir = 'data/case30_ieee'
    try:
        config_path = "config/"+data_dir.replace('data/','')+"base.yaml"
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    except:
        config_path = "config/case30_ieee_base.yaml"
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    config_args = NestedNamespace(**config_dict)
    node_normalizer, edge_normalizer = (
        BaseMVANormalizer(node_data=True,args=config_args),
        BaseMVANormalizer(node_data=False,args=config_args),
    )
    if randomMask:
        dataset = GridDatasetDisk(
            root=data_dir,
            norm_method="baseMVAnorm",
            node_normalizer=node_normalizer,
            edge_normalizer=edge_normalizer,
            pe_dim=20,  # Dimension of positional encoding
            transform=AddRandomMask(mask_dim=6,mask_ratio=0.5,args=config_args), #plus additional masks
        )
    else:
        dataset = GridDatasetDisk(
            root=data_dir,
            norm_method="baseMVAnorm",
            node_normalizer=node_normalizer,
            edge_normalizer=edge_normalizer,
            pe_dim=20,  # Dimension of positional encoding
            transform=AddPFMask(args=config_args),
        )

    # The scenarios are grouped in batches
    loader = DataLoader(dataset, batch_size=32)

    model.eval()
    if randomMask:
        diff = [[] for i in repeat(None, 18)]
        diffVar = [[] for i in repeat(None, 18)] 
    else:
        diff = [[] for i in repeat(None, 6)]
        diffVar = [[] for i in repeat(None, 6)] 

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)

            mask_PQ = batch.x[:, PQ] == 1 #PQ etc are just indexes
            mask_PV = batch.x[:, PV] == 1
            mask = batch.mask
            mask_REF = batch.x[:, REF] == 1
            # Apply random masking - not needed anymore due to new version of library
            #mask_value_expanded = model.mask_value.expand(batch.x.shape[0], -1)
            #batch.x[:, : batch.mask.shape[1]][batch.mask] = mask_value_expanded[batch.mask]
            #print(model.get_device())
            if topologyOnly:
                if ones:
                    replacement = torch.ones(batch.x.shape[0],6)
                else:
                    replacement = torch.zeros(batch.x.shape[0],6)
                batch.x[:,:6]=replacement
                #print(batch.x[:2,:]) #tested, works
                # Perform inference
                output = model(
                    batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch
                )
            else:
                # Perform inference
                output = model(
                    batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch
                )
            #filter out only nodes that it should predict
            ##iterate over all graphs and compute var/mean over prediction
            #print(output.shape) #960x6
            counter=0
            if torch.cuda.is_available():
                output = output.cpu()
                mask_PV = mask_PV.cpu()
                mask_PQ = mask_PV.cpu()
                batch = batch.cpu()
                mask_REF = mask_REF.cpu()
                mask = mask.cpu()
            if graph_wise:
                while counter in batch.batch:
                    if error_based:
                        error = np.sqrt((output[batch.batch==counter]-batch.y[batch.batch==counter])**2)
                    else:
                        error = output[batch.batch==counter]
                    if randomMask:    
                        combined_mask = torch.logical_and(mask_PQ[batch.batch==counter],mask[batch.batch==counter])
                        var, mean = torch.var_mean(error[combined_mask],dim=0)
                        for i, j in zip([6,7,8,9], [0,1,2,3]):
                            addNanCorrect(diff[i], mean[j].item(), removeNan)
                            addNanCorrect(diffVar[i], var[j].item(), removeNan)
                    else:
                        var, mean = torch.var_mean(error[mask_PQ[batch.batch==counter]],dim=0)
                    addNanCorrect(diff[0], mean[4].item(), removeNan)
                    addNanCorrect(diffVar[0], var[4].item(), removeNan)
                    addNanCorrect(diff[1], mean[5].item(), removeNan)
                    addNanCorrect(diffVar[1], var[5].item(), removeNan)
                    if randomMask:    
                        combined_mask = torch.logical_and(mask_PV[batch.batch==counter],mask[batch.batch==counter])
                        var, mean = torch.var_mean(error[combined_mask],dim=0)
                        for i, j in zip([10,11,12,13], [0,1,2,4]):
                            addNanCorrect(diff[i], mean[j].item(), removeNan)
                            addNanCorrect(diffVar[i], var[j].item(), removeNan)
                    else:
                        var, mean = torch.var_mean(error[mask_PV[batch.batch==counter]],dim=0)
                    addNanCorrect(diff[2], mean[3].item(), removeNan)
                    addNanCorrect(diffVar[2], var[3].item(), removeNan)
                    addNanCorrect(diff[3], mean[5].item(), removeNan)
                    addNanCorrect(diffVar[3], var[5].item(), removeNan)
                    if randomMask:    
                        combined_mask = torch.logical_and(mask_REF[batch.batch==counter],mask[batch.batch==counter])
                        var, mean = torch.var_mean(error[combined_mask],dim=0)
                        for i, j in zip([14,15,16,17], [0,1,4,5]):
                            addNanCorrect(diff[i], mean[j].item(), removeNan)
                            addNanCorrect(diffVar[i], var[j].item(), removeNan)
                    else:
                        var, mean = torch.var_mean(error[mask_REF[batch.batch==counter]],dim=0)
                    addNanCorrect(diff[4], mean[2].item(), removeNan)
                    addNanCorrect(diffVar[4], var[2].item(), removeNan)
                    addNanCorrect(diff[5], mean[3].item(), removeNan)
                    addNanCorrect(diffVar[5], var[3].item(), removeNan)
                    counter = counter+1 
            elif nodeWise: #compute per node
                if error_based:
                    error = np.sqrt((output-batch.y)**2)
                else:
                    error = output
                ###PQ
                if randomMask:
                    combined_mask = torch.logical_and(mask_PQ,mask)
                    tempRes=torch.nan_to_num(copy.copy(error[combined_mask,:]))
                    for i, j in zip([6,7,8,9], [0,1,2,3]):
                        diff[i]=diff[i]+tempRes[:,j].tolist()
                else:
                    tempRes = torch.nan_to_num(copy.copy(error[mask_PQ,:]))
                    diff[0]= diff[0]+tempRes[:,4].tolist()
                    diff[1] = diff[1]+tempRes[:,5].tolist()
                #### PV
                if randomMask:
                    combined_mask = torch.logical_and(mask_PV,mask)
                    tempRes=torch.nan_to_num(copy.copy(error[combined_mask,:]))
                    for i, j in zip([10,11,12,13], [0,1,2,4]):
                        diff[i]=diff[i]+tempRes[:,j].tolist()
                else: 
                    tempRes = torch.nan_to_num(copy.copy(error[mask_PV,:]))
                    diff[2]= diff[2]+tempRes[:,3].tolist()
                    diff[3] = diff[3]+tempRes[:,5].tolist()
                #REF
                if randomMask:
                    combined_mask = torch.logical_and(mask_REF,mask)
                    tempRes=torch.nan_to_num(copy.copy(error[combined_mask,:]))
                    for i, j in zip([14,15,16,17], [0,1,4,5]):
                        diff[i]=diff[i]+tempRes[:,j].tolist()
                else:
                    var, mean = torch.var_mean(error[mask_REF,:],dim=0) 
                    diff[4]= diff[4]+tempRes[:,2].tolist()
                    diff[5] = diff[5]+tempRes[:,3].tolist() 
            else: #compute per batch
                if error_based:
                    error = np.sqrt((output-batch.y)**2)
                else:
                    error = output
                #PQ
                if randomMask:
                    combined_mask = torch.logical_and(mask_PQ,mask)
                    var, mean = torch.var_mean(error[combined_mask,:],dim=0)
                    for i, j in zip([6,7,8,9], [0,1,2,3]):
                        addNanCorrect(diff[i], mean[j].item(), removeNan)
                        addNanCorrect(diffVar[i], var[j].item(), removeNan)
                else:
                    var, mean = torch.var_mean(error[mask_PQ,:],dim=0)
                addNanCorrect(diff[0], mean[4].item(), removeNan)
                addNanCorrect(diffVar[0], var[4].item(), removeNan)
                addNanCorrect(diff[1], mean[5].item(), removeNan)
                addNanCorrect(diffVar[1], var[5].item(), removeNan)
                #PV
                if randomMask:
                    combined_mask = torch.logical_and(mask_PV,mask)
                    var, mean = torch.var_mean(error[combined_mask,:],dim=0)
                    for i, j in zip([10,11,12,13], [0,1,2,4]):
                        addNanCorrect(diff[i], mean[j].item(), removeNan)
                        addNanCorrect(diffVar[i], var[j].item(), removeNan)
                else:
                    var, mean = torch.var_mean(error[mask_PV,:],dim=0)   
                addNanCorrect(diff[2], mean[3].item(), removeNan)
                addNanCorrect(diffVar[2], var[3].item(), removeNan)
                addNanCorrect(diff[3], mean[5].item(), removeNan)
                addNanCorrect(diffVar[3], var[5].item(), removeNan)  
                #REF
                if randomMask:
                    combined_mask = torch.logical_and(mask_REF,mask)
                    var, mean = torch.var_mean(error[combined_mask,:],dim=0)
                    for i, j in zip([14,15,16,17], [0,1,4,5]):
                        addNanCorrect(diff[i], mean[j].item(), removeNan)
                        addNanCorrect(diffVar[i], var[j].item(), removeNan)
                else:
                    var, mean = torch.var_mean(error[mask_REF,:],dim=0)  
                addNanCorrect(diff[4], mean[2].item(), removeNan)
                addNanCorrect(diffVar[4], var[2].item(), removeNan)
                addNanCorrect(diff[5], mean[3].item(), removeNan)
                addNanCorrect(diffVar[5], var[3].item(), removeNan)  
    #for i in range(len(diff)):
    #    print(len(diff[i]),len(diffVar[i]))
    return diff, diffVar  

def evalMIallFeatures(seenData, unseenData,titles, suffix, blackbox=True,threads=4,leaveOneOut=False,modelComparison=False):
    if leaveOneOut:
        MembershipBBMinusOneEval(seenData,unseenData, 'All_Loo', suffix, dim=len(seenData[0])) 
    elif modelComparison:
        ClassifierAblationMembership(seenData,unseenData, 'All', suffix, dim=len(seenData[0]))
    else:
        MembershipBlackBox(seenData,unseenData, 'All', suffix, dim=len(seenData[0])) 

def evalMI(seenData, unseenData,titles, suffix, blackbox=True,threads=4, leaveOneOut=False,modelComparison=True):
    #iters = int(((len(titles)+1.0)/threads)+0.9)
    #tempind = np.array_split(np.arange(len(titles)),iters)
    #print(iters, tempind, len(seenData),len(seenData[0]))
    #for index in range(iters):
    #    with Pool(threads) as p:
    #        params = list(zip(seenData[tempind[index]], unseenData[tempind[index]], np.array(titles)[tempind[index]], [suffix]*np.shape(tempind[index])[0]))
    #        p.starmap(MembershipBlackBox, params)
    #        #p.starmap(debugMIDummy, params)
    for i in range(len(titles)):
        MembershipBlackBox(seenData[i],unseenData[i],titles[i],suffix,modelComparison=True)


def fitSVM(dataO, labels,maxSamples=-1,fullComparison=False):
    data=copy.copy(dataO)
    largeData=False
    if maxSamples>0:
        # set maximum amount of samples to this
        #print('fit SMM max samples',maxSamples,np.shape(data)[0])
        if np.shape(data)[0]<=maxSamples:
            X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42) 
        else:   
            #print(float(maxSamples),float(np.shape(data)[0]),(float(maxSamples)/float(np.shape(data)[0])))
            perTrain = float(maxSamples)/float(np.shape(data)[0])
            perTest = np.min([15000.0/float(np.shape(data)[0]),(1.0-perTrain)])
            print(perTrain, perTest)
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=perTest, train_size=perTrain, random_state=42)
            print(np.shape(X_train),np.shape(X_test))
            if np.shape(X_train)[0]>70000:
                #if we're still to large we cap as before
                largeData= True   
    if (maxSamples <0 and np.shape(data)[0]>70000) or largeData:
        #print('running also big data')
        #for runtime reasons, we only train on max 90,000 samples
        # per = 70000.0/float(np.shape(data)[0]) #previously
        per = 1.0-(50000.0/float(np.shape(data)[0]))
        X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=per, random_state=42)
        #we also don't test on all remaining samples
        per = 15000.0/float(np.shape(X_temp)[0])
        _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=per, random_state=42)
    elif maxSamples <0 and np.shape(data)[0]<=70000:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    #print('should be as above',np.shape(X_train))    
    bestPerf = 0.0
    bestclf = None
    bestkernel = ''
    bestc = 0
    bestgamma = 0
    #print(np.shape(X_train),np.shape(X_test))
    for c in  [0.01, 0.1, 1, 10, 100]: #removed 0.001
        print('linear',c)
        clf = svm.SVC(C=c, kernel='linear')
        clf.fit(X_train,y_train)
        res = clf.score(X_test,y_test)
        if res > bestPerf:
            bestPerf = res
            bestclf = clf
            bestc =c
            bestgamma = -1.0
            bestkernel = 'linear'
    for gamma in [0.001, 0.01, 0.1, 1,10,100]: #removed 0.0001 due to poor performance
        for c in  [0.01, 0.1, 1, 10, 100]: #removed 0.001
            print('rbf',c,gamma)
            clf = svm.SVC(C=c, kernel='rbf', gamma=gamma)
            clf.fit(X_train,y_train)
            res = clf.score(X_test,y_test)
            if res > bestPerf:
                bestPerf = res
                bestclf = clf
                bestc =c
                bestgamma = gamma
                bestkernel = 'rbf'
    #print(np.shape(X_train),np.shape(X_test),bestPerf)
    if fullComparison:
        for depth in [5,10,15,20,25]:
            print('random forest',depth)
            clf = RandomForestClassifier(max_depth=depth, random_state=0)
            clf.fit(X_train,y_train)
            res = clf.score(X_test,y_test)
            if res > bestPerf:
                bestPerf = res
                bestclf = clf
                bestc = 'max_depth '+str(depth)
                bestgamma = ''
                bestkernel = 'random forest'
        '''X_train_n=copy.copy(X_train)
        X_test_n=copy.copy(X_test)
        max_v = np.max(X_train_n)
        min_v = np.min(X_train_n)
        range_v = max_v - min_v
        if range_v == 0:
            raise ValueError("Training data has zero variance; min == max.")
        # Apply the same transform to train and test
        X_train_n = (X_train_n - min_v) / range_v
        X_test_n  = (X_test_n  - min_v) / range_v

        for width in [25,50,100,500]:
            for lr in [1e-5,1e-4,1e-3]:

                clf = MLPClassifier(solver='adam', learning_rate_init=lr,hidden_layer_sizes=(width, width), random_state=1, max_iter=500)
                clf.fit(X_train_n,y_train)
                res = clf.score(X_test_n,y_test)
                if res > bestPerf:
                    bestPerf = res
                    bestclf = clf
                    bestc = 'width '+str(width)
                    bestgamma = 'lr = '+str(lr)
                    bestkernel = 'MLP' '''
        #add unsupervised comaprison
        print('kmeans')
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_train)
        kmeans_labels = kmeans.predict(X_test)
        res = adjusted_rand_score(y_test, kmeans_labels)
        if res > bestPerf:
            bestPerf = res
            bestclf = kmeans
            bestc = '2 '
            bestgamma = '2 '
            bestkernel = 'K-means'
        #for eps in [0.01, 0.1, 0.5, 1.0, 5.0]:
        #    dbscan = DBSCAN(eps=0.5, min_samples=5)
        #    dbscan.fit(X_train)
        #    dbscan_labels = dbscan.predict(X_test)
        #    res = adjusted_rand_score(y_test, dbscan_labels)
        #    if res > bestPerf:
        #        bestPerf = res
        #        bestclf = kmeans
        #        bestc = str(eps)
        #        bestgamma = '2 clusters'
        #        bestkernel = 'DBScan'
    return bestclf, bestPerf, 'c '+str(bestc), bestgamma, bestkernel

def recombineLeaveOut(arr, dim=1,index=1):
    if dim==1:
        combined = []
        c = 0
        for elem in arr:
            if c!=index:
                combined = combined+elem
            c = c+1
    else:
        combined = [[] for i in repeat(None, dim)]
        for i in range(dim):
            for j in range(len(arr[i])):
                if index!=j:
                    combined[i]=combined[i]+arr[i][j]
    return combined   

def recombine(arr, dim=1,index=-1):
    if dim==1:
        combined = []
        for elem in arr:
            combined = combined+elem
    else:
        combined = [[] for i in repeat(None, dim)]
        if index>=0:
            for i in range(dim):
                combined[i]=combined[i]+arr[i][index]
        else:
            for i in range(dim):
                for elem in arr[i]:
                    combined[i]=combined[i]+elem
    return combined

def MembershipAnalysisAblation(seenData, unseenData, suffix):
    dataSizes = [50000,25000,12500,6250,3125,1562,781,390,195,85,42,21]
    scores = []
    baselineAcc = 0.0
    fass = []
    cs = []
    gammas = []
    kernels = []
    for size in dataSizes:
        fullScore, baseline, fas, c, gamma, kernel = SimpleMembershipBlackBox(seenData,unseenData, 'All', suffix, dim=len(seenData[0]),maxSamples=size) 
        scores.append(fullScore)
        baselineAcc = baseline #remains the same, suffices to save once
        fass.append(fas)
        cs.append(c)
        gammas.append(gamma)
        kernels.append(kernel)
    #print SVM params to check consistency
    print(cs, gammas, kernels)
    print(scores)
    print(dataSizes)
    #plot remainder to understand effect of known amount of data
    plt.style.use('ggplot')
    fig = plt.figure()
    plt.hlines(baselineAcc,0,len(scores),colors=['k'])
    plt.plot(dataSizes,scores,c=colors[0])
    plt.plot(dataSizes,fass,c=colors[-1])
    plt.xlabel('Number of Samples')
    plt.ylabel('Performance')
    plt.savefig('plots/MI/AblationAll_'+suffix+'.pdf')
    plt.clf()
    plt.cla()
    res = np.zeros((2,len(dataSizes)))
    res[0,:]=scores
    res[1,:]=fass
    np.save(plot_directory+'files/AblationAll_'+suffix+'.npy',res)

def SimpleMembershipBlackBox(seenData,unseenData, title, suffix, dim=1,maxSamples=-1):
    # on full data - worst case / theoretically achievable
    fulltrain = recombine(seenData,dim)
    fullTest = recombine(unseenData,dim)
    if len(fulltrain)==0 or len(fullTest)==0:
        return None
    #compute over all datasets
    # we know the labels, as seen / unseen corresponds to label. we gernate them on the fly
    if dim==1:
        labs = getLabels(len(fulltrain),len(fullTest))
        data = np.array(fulltrain+fullTest).reshape(-1,dim)
    else:
        data = np.hstack((np.array(fulltrain),np.array(fullTest)))
        data = np.transpose(data)
        labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
    clf, fullScore, c, gamma, kernel = fitSVM(data, labs,maxSamples,fullComparison=True)
    joblib.dump(clf, temporary_storage+'SVM/Abl'+str(maxSamples)+title+suffix+'_SVM.pickle')
    baselineAcc = float(np.mean(labs))
    fas = getFalseAlarmRate(clf,data,labs)
    return fullScore, baselineAcc, fas, c, gamma, kernel


def MembershipBBMinusOneEval(seenData,unseenData, title, suffix, dim=1,maxSamples=-1,modelComparison=True):
    suffix = 'n-1_'+suffix
    if dim==1:
        raise NotImplementedError
    else:
        fulltrain = recombine(seenData,dim)
        fullTest = recombine(unseenData,dim)
        if len(fulltrain)==0 or len(fullTest)==0:
            return None
        #compute over all datasets
        # we know the labels, as seen / unseen corresponds to label. we gernate them on the fly
        data = np.hstack((np.array(fulltrain),np.array(fullTest)))
        data = np.transpose(data)
        labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
        clf, fullScore, c, gamma, kernel = fitSVM(data, labs,maxSamples,fullComparison=modelComparison)
        joblib.dump(clf, temporary_storage+'SVM/'+title+suffix+'_SVM.pickle')
        baselineAcc = float(np.mean(labs))
        fas = getFalseAlarmRate(clf,data,labs)
        res = DataStruct(fullScore, baselineAcc, fas, c, gamma, kernel)
        for i in range(len(seenData[0])):
            testSeen = recombine(seenData,dim,i)
            #this is the test data
            fulltrain = recombineLeaveOut(seenData,dim,i)
            ##this is the training data, just drop the one used for testing
            for j in range(len(unseenData[0])):
                testUnseen = recombine(unseenData,dim,j)
                ### this is our test data
                fullTest = recombineLeaveOut(unseenData,dim,j)
                ##this is the training data, just drop the one used for testing
                if len(fulltrain)==0 or len(fullTest)==0:
                    return None
                data = np.hstack((np.array(fulltrain),np.array(fullTest)))
                data = np.transpose(data)
                labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
                #train SVM
                clf, fullScore, c, gamma, kernel = fitSVM(data, labs,maxSamples,fullComparison=modelComparison)
                indRes = Result(fullScore, c, gamma, kernel, i, j)
                #now evaluate on all other datasets that we didn't use for the SVM
                parScore = clf.score(np.array(testSeen).reshape(-1,dim),getLabels(np.shape(testSeen)[1],0))
                indRes.addSeenResult(parScore)
                parScore = clf.score(np.array(testUnseen).reshape(-1,dim),getLabels(np.shape(testUnseen)[1],0))
                fas = getFalseAlarmRate(clf,np.array(testUnseen).reshape(-1,dim),getLabels(np.shape(testUnseen)[1],0))
                indRes.addUnseenResult(parScore)
                indRes.addFASResult(fas)
                res.addResult(indRes)
    # write output setting
    print(title, suffix)
    print(res.settingsStats())
    res.summarize(name=title+suffix, printRes=True)
    # save plot to analyse performance
    res.plot(title, suffix+'_MI',plot_directory)
    with open(temporary_storage+'interRes/'+title+suffix+'MI.pickle', 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    #res.serialize(temporary_storage+'interRes/',title,suffix+'MI.pickle')

def ClassifierAblationMembership(seenData,unseenData, title, suffix, dim=1):
    suffix = suffix+'_FMCaUS'
    # on full data - worst case / theoretically achievable
    fulltrain = recombine(seenData,dim)
    fullTest = recombine(unseenData,dim)
    if len(fulltrain)==0 or len(fullTest)==0:
        return None
    #compute over all datasets
    # we know the labels, as seen / unseen corresponds to label. we gernate them on the fly
    if dim==1:
        labs = getLabels(len(fulltrain),len(fullTest))
        data = np.array(fulltrain+fullTest).reshape(-1,dim)
    else:
        data = np.hstack((np.array(fulltrain),np.array(fullTest)))
        data = np.transpose(data)
        labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
    clf, fullScore, c, gamma, kernel = fitSVM(data, labs,-1,fullComparison=True)
    baselineAcc = float(np.mean(labs))
    fas = getFalseAlarmRate(clf,data,labs)
    res = DataStruct(fullScore, baselineAcc, fas, c, gamma, kernel)
    # above, we trained on a subset of everything. Now, we use only
    # two datasets and test how the SVM generalizes on the remaining things
    if True:
        for i in range(len(seenData[0])):
            for j in range(len(unseenData[0])):
                trainD = recombine(seenData,dim,i)
                testD = recombine(unseenData,dim,j)
                #print(np.shape(trainD),np.shape(testD))
                data = np.hstack((np.array(trainD),np.array(testD)))
                data = np.transpose(data)
                labs = getLabels(np.shape(trainD)[1],np.shape(testD)[1])
                #train SVM
                clf, fullScore, c, gamma, kernel = fitSVM(data, labs,-1,fullComparison=True)
                indRes = Result(fullScore, c, gamma, kernel, i, j)
                #now evaluate on all other datasets that we didn't use for the SVM
                for k in range(len(seenData[0])):
                    if k==i:
                        pass       
                    else:
                        trainTemp = recombine(seenData,dim,k)
                        parScore = clf.score(np.array(trainTemp).reshape(-1,dim),getLabels(np.shape(trainTemp)[1],0))
                        indRes.addSeenResult(parScore)
                for k in range(len(unseenData[0])):
                    if k==j:
                        pass
                    else:
                        testTemp = recombine(unseenData,dim,k)
                        parScore = clf.score(np.array(testTemp).reshape(-1,dim),getLabels(np.shape(testTemp)[1],0))
                        fas = getFalseAlarmRate(clf,np.array(testTemp).reshape(-1,dim),getLabels(np.shape(testTemp)[1],0))
                        indRes.addUnseenResult(parScore)
                        indRes.addFASResult(fas)
                res.addResult(indRes)
    # write output setting
    print(title, suffix)
    print(res.settingsStats())
    res.summarize(name=title+suffix, printRes=True)
    # save plot to analyse performance
    res.plot(title, suffix+'_MI',plot_directory)
    with open(temporary_storage+'interRes/'+title+suffix+'MI.pickle', 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    #res.serialize(temporary_storage+'interRes/',title,suffix+'MI.pickle')


def MembershipBlackBox(seenData,unseenData, title, suffix, dim=1,maxSamples=-1,modelComparison=True):
    if maxSamples>0:
        suffix = suffix+'S'+str(maxSamples)
    # on full data - worst case / theoretically achievable
    fulltrain = recombine(seenData,dim)
    fullTest = recombine(unseenData,dim)
    if len(fulltrain)==0 or len(fullTest)==0:
        return None
    #compute over all datasets
    # we know the labels, as seen / unseen corresponds to label. we gernate them on the fly
    if dim==1:
        labs = getLabels(len(fulltrain),len(fullTest))
        data = np.array(fulltrain+fullTest).reshape(-1,dim)
    else:
        data = np.hstack((np.array(fulltrain),np.array(fullTest)))
        data = np.transpose(data)
        labs = getLabels(np.shape(np.array(fulltrain))[1],np.shape(np.array(fullTest))[1])
    clf, fullScore, c, gamma, kernel = fitSVM(data, labs,maxSamples,fullComparison=modelComparison)
    joblib.dump(clf, temporary_storage+'SVM/'+title+suffix+'_SVM.pickle')
    baselineAcc = float(np.mean(labs))
    fas = getFalseAlarmRate(clf,data,labs)
    res = DataStruct(fullScore, baselineAcc, fas, c, gamma, kernel)
    # above, we trained on a subset of everything. Now, we use only
    # two datasets and test how the SVM generalizes on the remaining things
    if dim==1:
        for i in range(len(seenData)):
            for j in range(len(unseenData)):
                print('within loop',len(seenData),len(seenData[i]),len(unseenData[j]))
                labs = getLabels(len(seenData[i]),len(unseenData[j]))
                data = np.array(seenData[i]+unseenData[j]).reshape(-1,dim) 
                clf, fullScore, c, gamma, kernel = fitSVM(data, labs,maxSamples,fullComparison=modelComparison)
                indRes = Result(fullScore, c, gamma, kernel, i, j)
                #now evaluate on all other datasets that we didn't use for the SVM
                for k in range(len(seenData)):
                    if k==i:
                        pass       
                    else:
                        parScore = clf.score(np.array(seenData[k]).reshape(-1,dim),getLabels(len(seenData[k]),0))
                        indRes.addSeenResult(parScore)
                for k in range(len(unseenData)):
                    if k==j:
                        pass
                    else:
                        parScore = clf.score(np.array(unseenData[k]).reshape(-1,dim),getLabels(0,len(unseenData[k])))
                        fas = getFalseAlarmRate(clf,np.array(unseenData[k]).reshape(-1,dim),getLabels(0,len(unseenData[k])))
                        indRes.addUnseenResult(parScore)
                        indRes.addFASResult(fas)
                res.addResult(indRes)
    else:
        for i in range(len(seenData[0])):
            for j in range(len(unseenData[0])):
                trainD = recombine(seenData,dim,i)
                testD = recombine(unseenData,dim,j)
                #print(np.shape(trainD),np.shape(testD))
                data = np.hstack((np.array(trainD),np.array(testD)))
                data = np.transpose(data)
                labs = getLabels(np.shape(trainD)[1],np.shape(testD)[1])
                #train SVM
                clf, fullScore, c, gamma, kernel = fitSVM(data, labs,maxSamples,fullComparison=modelComparison)
                indRes = Result(fullScore, c, gamma, kernel, i, j)
                #now evaluate on all other datasets that we didn't use for the SVM
                for k in range(len(seenData[0])):
                    if k==i:
                        pass       
                    else:
                        trainTemp = recombine(seenData,dim,k)
                        parScore = clf.score(np.array(trainTemp).reshape(-1,dim),getLabels(np.shape(trainTemp)[1],0))
                        indRes.addSeenResult(parScore)
                for k in range(len(unseenData[0])):
                    if k==j:
                        pass
                    else:
                        testTemp = recombine(unseenData,dim,k)
                        parScore = clf.score(np.array(testTemp).reshape(-1,dim),getLabels(np.shape(testTemp)[1],0))
                        fas = getFalseAlarmRate(clf,np.array(testTemp).reshape(-1,dim),getLabels(np.shape(testTemp)[1],0))
                        indRes.addUnseenResult(parScore)
                        indRes.addFASResult(fas)
                res.addResult(indRes)
    # write output setting
    print(title, suffix)
    print(res.settingsStats())
    res.summarize(name=title+suffix, printRes=True)
    # save plot to analyse performance
    res.plot(title, suffix+'_MI',plot_directory)
    with open(temporary_storage+'interRes/'+title+suffix+'MI.pickle', 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    #res.serialize(temporary_storage+'interRes/',title,suffix+'MI.pickle')

# This network was chosen for visualization purposes (networks up to 300 buses were tested)
# The number of load scenarios is 1024

def debugMI(graphWise, model12, randomMask, topologyOnly, featureNames):
    if model12:
        model = loadModel("models/GridFM_v0_1.pth")
    else:
        model = loadModel("models/GridFM_v0_2.pth")
    #non-training data
    if randomMask:
        unseenMean = [[] for i in repeat(None, 18)]
        unseenVar = [[] for i in repeat(None, 18)]  
    else:
        unseenMean = [[] for i in repeat(None, 6)]
        unseenVar = [[] for i in repeat(None, 6)] 
    mean, var = genData('case14_ieee', model,graph_wise=graphWise,randomMask=randomMask)
    #print('generated data',len(mean),len(mean[0]))
    for i in range(len(mean)):
        unseenMean[i].append(mean[i])
        unseenMean[i].append(mean[i])
    if randomMask:
        seenMean = [[] for i in repeat(None, 18)]
        seenVar = [[] for i in repeat(None, 18)] 
    else:
        seenMean = [[] for i in repeat(None, 6)]
        seenVar = [[] for i in repeat(None, 6)] 
    mean, var = genData('case30_ieee', model,graph_wise=graphWise,randomMask=randomMask)
    #print('generated train data',len(mean),len(mean[0]))
    for i in range(len(mean)):
        seenMean[i].append(mean[i])
        seenMean[i].append(mean[i])
    evalMI(unseenMean,seenMean,['14','14','30','30'], 'DEBUG2')

def genCasesSVMCrossEval(case='mean', featureList=[]):
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


    

def mainExperiments(graphWise,nodeWise, model12, randomMask, topologyOnly, featureNames, removeNan, plot, MI, MIAblation=False, runIndividualFeatures=False,step='load',data_dir_seen=[],data_dir_unseen=[],error_based=True,leaveOneOut=False, modelComparison=False, ones=False):
    #print('started',graphWise,nodeWise, model12,topologyOnly)
    print('ones and topology, should be TRUE: ', topologyOnly,ones)
    data_dir = temporary_storage
    unseenDataNames = ['39 epri','60c','1354','197 snem','300','73 rts','14','5 pfm']
    seenNames = ['240 pserc','24 ieee','57 ieee','89 pegase','118 ieee','30']
    suffix = ''
    if graphWise:
        suffix = suffix+'GW_'
    if nodeWise:
        suffix = suffix+'NW_'
    if model12:
        suffix = suffix+'M12_'
    if randomMask:
        suffix = suffix+'R_'
    if topologyOnly:
        if ones:
            print('testing ones!')
            suffix = suffix+'topolOnlyOnes_'
        else:
            suffix = suffix+'topolOnly_'
    if not removeNan:
        suffix = suffix+'NaNZ_'
    if not error_based:
        suffix = suffix+'nonErr_'
    print('should contain ones ',suffix)
    if 'gen' in step:        ### check whether data is already there
        try:
            with open(data_dir+featureNames[0]+suffix+'meanTrainTemp.pickle', 'rb') as f:
                _ = pickle.load(f)
            #step = 'schr'
            print(suffix,' Data already generated!')
        except FileNotFoundError:
            pass
    if 'gen' in step:
        if model12:
            model = loadModel("models/GridFM_v0_1.pth")
        else:
            model = loadModel("models/GridFM_v0_2.pth")
        if torch.cuda.is_available():
            #move model to gpu
            model.model.to('cuda')
        if randomMask:
            unseenMean = [[] for i in repeat(None, 18)]
            unseenVar = [[] for i in repeat(None, 18)] 
        else:
            unseenMean = [[] for i in repeat(None, 6)]
            unseenVar = [[] for i in repeat(None, 6)]
        if len(data_dir_unseen)<1:
            data_dirs = ['case39_epri','case60_c','case1354_pegase','case197_snem','case300_ieee','case73_ieee_rts','case14_ieee','case5_pjm']
        else:
            data_dirs = data_dir_unseen
        for datadir in data_dirs:
            mean, var = genData(datadir, model,graph_wise=graphWise,nodeWise=nodeWise,randomMask=randomMask,topologyOnly=topologyOnly, removeNan=removeNan, error_based=error_based)
            for i in range(len(mean)):
                unseenMean[i].append(mean[i])
                if not nodeWise:
                    unseenVar[i].append(var[i])
        if len(data_dir_seen)<1:
            data_dirs = ['case240_pserc','case24_ieee_rts', 'case57_ieee','case89_pegase','case118_ieee','case30_ieee']
        else:
            data_dirs = data_dir_seen
        if randomMask:
            seenMean = [[] for i in repeat(None, 18)]
            seenVar = [[] for i in repeat(None, 18)] 
        else:
            seenMean = [[] for i in repeat(None, 6)]
            seenVar = [[] for i in repeat(None, 6)]
        for datadir in data_dirs:
            mean, var = genData(datadir, model,graph_wise=graphWise,nodeWise=nodeWise,randomMask=randomMask,topologyOnly=topologyOnly, removeNan=removeNan,error_based=error_based)
            for i in range(len(mean)):
                seenMean[i].append(mean[i])
                if not nodeWise:
                    seenVar[i].append(var[i])
        #serialize
        if 'serial' in step:
            #train mean, train var, test mean, test var
            maxIter = 6
            if randomMask: 
                maxIter=18
            data_dir = temporary_storage
            for i in range(maxIter):
                print(data_dir)
                print('saving',data_dir+featureNames[i]+suffix+'meanTrainTemp.pickle')
                with open(data_dir+featureNames[i]+suffix+'meanTrainTemp.pickle', 'wb') as f:
                    # Pickle the arrays using the highest protocol available.
                    pickle.dump(seenMean[i], f, pickle.HIGHEST_PROTOCOL)
                with open(data_dir+featureNames[i]+suffix+'meanEvalTemp.pickle', 'wb') as f:
                    # Pickle the arrays using the highest protocol available.
                    pickle.dump(unseenMean[i], f, pickle.HIGHEST_PROTOCOL)
                if not nodeWise:
                    with open(data_dir+featureNames[i]+suffix+'varTrainTemp.pickle', 'wb') as f:
                        # Pickle the arrays using the highest protocol available.
                        pickle.dump(seenVar[i], f, pickle.HIGHEST_PROTOCOL)
                    with open(data_dir+featureNames[i]+suffix+'varEvalTemp.pickle', 'wb') as f:
                        # Pickle the arrays using the highest protocol available.
                        pickle.dump(unseenVar[i], f, pickle.HIGHEST_PROTOCOL)
    #desearialize
    if 'load' in step:
        seenMean = []
        seenVar = []
        unseenMean = []
        unseenVar = []
        maxIter = 6
        if randomMask: 
            maxIter=18
        try:
            data_dir = temporary_storage
            for i in range(maxIter):
                with open(data_dir+featureNames[i]+suffix+'meanTrainTemp.pickle', 'rb') as f:
                    seenMean.append(pickle.load(f))
                with open(data_dir+featureNames[i]+suffix+'meanEvalTemp.pickle', 'rb') as f:
                    unseenMean.append(pickle.load(f))
                if not nodeWise:
                    with open(data_dir+featureNames[i]+suffix+'varEvalTemp.pickle', 'rb') as f:
                        unseenVar.append(pickle.load(f))
                    with open(data_dir+featureNames[i]+suffix+'varTrainTemp.pickle', 'rb') as f:
                        seenVar.append(pickle.load(f))
            #sanity check Length - not neccessary, loading for feature would fail
            #print('sanity check data loading')
            #for i in range(maxIter):
            #    print(len(seenMean[i]))
            #    print(len(unseenMean[i]))
        except FileNotFoundError:
            print(featureNames[i]+suffix, 'NOT FOUND!!')
            plot = False
            MI = False
    #adding this as it is only relevant for MI
    if modelComparison:
        suffix = suffix+'FMC'
    if plot:
        plotAll(seenMean,unseenMean,seenNames, unseenDataNames, featureNames, suffix+'mean')
        if not nodeWise:
            plotAll(seenVar,unseenVar,seenNames, unseenDataNames, featureNames, suffix+'var')
    if MIAblation:
        MembershipAnalysisAblation(seenMean, unseenMean, suffix+'mean')
        MembershipAnalysisAblation(seenVar, unseenVar, suffix+'var')
    if MI:
        #check membership
        if not removeNan and not runIndividualFeatures:
            evalMIallFeatures(seenMean,unseenMean, featureNames, suffix+'mean',leaveOneOut=leaveOneOut,modelComparison=modelComparison)
            if not nodeWise:
                evalMIallFeatures(seenVar,unseenVar, featureNames, suffix+'var',leaveOneOut=leaveOneOut,modelComparison=modelComparison)
        if runIndividualFeatures:
            evalMI(seenMean,unseenMean, featureNames, suffix+'mean',leaveOneOut=leaveOneOut,modelComparison=modelComparison)
            if not nodeWise:
                evalMI(seenVar,unseenVar, featureNames, suffix+'var',leaveOneOut=leaveOneOut,modelComparison=modelComparison) 

