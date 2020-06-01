
# Copyright (C) 2020 Yegor Tkachenko, Kamel Jedidi
# Code -- Study 2 -- What Personal Information Can a Consumer Facial Image Reveal?
# https://github.com/computationalmarketing/facialanalysis/

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib import gridspec

from matplotlib import rcParams
rcParams.update({'font.size': 12})

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']

import seaborn as sns

import torchvision.models as models
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import os
from os import walk
from tqdm import tqdm

from sklearn.utils import class_weight
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import KFold, GroupKFold, ShuffleSplit, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

import scipy.stats
from scipy.special import softmax
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage

# ATTENTION: we disable notifications when AUC cannot be computed -- during nn finetuning
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

import json

import numpy as np

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd

import pickle

import sys


'''
CustomDataset object takes care of supplying an observation (image, labels).
It also performs image preprocessing, such as normalization by color channel. 
In case of training, it also performs random transformations, such as horizontal flips, resized crops, rotations, and color jitter.
'''


class CustomDataset(Dataset):

    def __init__(self, data, tr = True):

        self.data = data 
        self.paths = self.data['img_path'].values.astype('str')
        self.data_len = self.data.shape[0]

        self.labels = self.data[q_list].values.astype('int32')
        self.control_metrics = self.data[control_list].values.astype('float32')

        # transforms
        if tr:
            self.transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1)], p=0.75),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(), 
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        
        img_path = PATH + '/'+ self.paths[index]
        img = Image.open(img_path)
        img_tensor = self.transforms(img)
        label = self.labels[index]
        control_metric = self.control_metrics[index]

        return (img_tensor, label, control_metric)

    def __len__(self):
        return self.data_len



#get pretrained resnet50 model
def get_pretrained():
    model = models.resnet50(pretrained=True)
    return model


# replace last layer
def prepare_for_finetuning(model):

    for param in model.parameters():
        param.requires_grad = False
        param.requires_grad = True

    #replacing last layer with new fully connected
    model.fc = torch.nn.Linear(model.fc.in_features,n_outs)
    return

# create an object that uses CustomDataset object from above to load multiple observations in parallel
def create_dataloader(data,rand=True):

    if rand: # shuddle observations
        dataset = CustomDataset(data, tr=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False)
    
    else: # load in fixed order of data
        dataset = CustomDataset(data, tr=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler = torch.utils.data.sampler.SequentialSampler(dataset), num_workers=10, drop_last=False)

    return loader



#finetune and save neural net model
def finetune_and_save(loader_train, loader_test):

    # loading pretrained model and preparing it for finetuning

    model = get_pretrained()
    prepare_for_finetuning(model)
    if CUDA:
        model.cuda()

    # optimize only last six layers
    layers = list(model.children())
    params = list(layers[len(layers)-1].parameters())+list(layers[len(layers)-2].parameters())+list(layers[len(layers)-3].parameters())+list(layers[len(layers)-4].parameters())+list(layers[len(layers)-5].parameters())+list(layers[len(layers)-6].parameters())
    optimizer = optim.Adamax(params=params, lr=0.001)

    # print("starting finetuning")
    hist = {}
    hist['d_labs'] = q_list

    hist['train_loss'] = []
    hist['val_loss'] = []

    hist['train_loss_d'] = []
    hist['val_loss_d'] = []

    hist['train_auc_d'] = []
    hist['val_auc_d'] = []

    acc_best = 0.0

    #train
    for epoch in range(N_EPOCHS):
        
        train_loss, train_loss_d, train_auc_d = run_epoch(model, loss_f, optimizer, loader_train, update_model = True) # training
        eval_loss, eval_loss_d, eval_auc_d = run_epoch(model, loss_f, optimizer, loader_test, update_model = False) # evaluation

        hist['train_loss'].append(train_loss)
        hist['val_loss'].append(eval_loss)

        hist['train_loss_d'].append(train_loss_d)
        hist['val_loss_d'].append(eval_loss_d)

        hist['train_auc_d'].append(train_auc_d)
        hist['val_auc_d'].append(eval_auc_d)

        with open(RESULTS+'/eval_record.json', 'w') as fjson:
            json.dump(hist, fjson)

    # saving model
    torch.save(model, RESULTS+"/finetuned_model")
    return



# function that performa training (or evaluation) over an epoch (full pass through a data set)
def run_epoch(model, loss_f, optimizer, loader, update_model = False):

    if update_model:
        model.train()
    else:
        model.eval()

    loss_hist = []
    loss_hist_detailed = []
    auc_hist_detailed = []

    for batch_i, var in tqdm(enumerate(loader)):

        loss, loss_detailed, auc_detailed = loss_f(model, var)

        if update_model:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_hist.append(loss.data.item())
        loss_hist_detailed.append(loss_detailed)
        auc_hist_detailed.append(auc_detailed)

    loss_detailed = pd.DataFrame(loss_hist_detailed)
    loss_detailed.columns = q_list

    auc_detailed = pd.DataFrame(auc_hist_detailed)
    auc_detailed.columns = q_list

    return np.mean(loss_hist).item(), loss_detailed.mean(0).values.tolist(), auc_detailed.mean(0).values.tolist()



# function to compute loss from a batch data
def loss_f(model, var):
    
    data, target, _ = var
    data, target = Variable(data), Variable(target)
    if CUDA:
        data, target = data.cuda(), target.cuda()
    
    output = model(data) # match for the user and focal game
    
    loss = 0
    loss_detailed = []
    auc_detailed = []

    for i in range(len(q_d_list)):

        w = torch.FloatTensor(class_weights[i])
        if CUDA:
           w = w.cuda()

        # output contains scores for each level of every predicted variable
        # q_d_list[i] is number of levels to variable i
        # q_d_list_cumsum[i] is a cumulative sum over number of levels for variable i and all variables before it
        # all variables ordered as in q_list
        # (q_d_list_cumsum[i]-q_d_list[i]):q_d_list_cumsum[i] then gives exact coordinates of the scores for variable i
        # among all scores in the output
        temp = F.cross_entropy(output[:,(q_d_list_cumsum[i]-q_d_list[i]):q_d_list_cumsum[i]], target[:,i].long(), weight=w)
        loss_detailed.append(temp.data.item())
        loss += temp

        # now we calculate AUC
        y_true = target[:,i].detach().cpu().numpy()
        y_score = output[:,(q_d_list_cumsum[i]-q_d_list[i]):q_d_list_cumsum[i]].detach().cpu().numpy()[:,1]

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc_detailed.append(metrics.auc(fpr, tpr))

    return loss, loss_detailed, auc_detailed

# building class balancing weights as in
# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
def calculate_class_weights(X):
    class_weights = []
    for i in q_list:
        class_weights.append(
            class_weight.compute_class_weight('balanced', np.unique(X[i].values), X[i].values))

    return class_weights

# extract data from a dataloader as a set of image features X and set of labels y, corresponding to those image features
# can also blackout specified areas of the loaded images before extracting the image features -- this is used in our experiments
# when data loader is deterministic, then it will load in the same data again and again
def extract_data(loader, modelred, blackout=None):

    X = []
    y = []
    z = []

    for batch_i, var in tqdm(enumerate(loader)):

        data, target, control_metrics = var
        
        if blackout is not None:
            data[:, :, blackout[0]:blackout[1],  blackout[2]:blackout[3]] = 0.0

        data, target, control_metrics = Variable(data), Variable(target), Variable(control_metrics)
        if CUDA:
            data, target, control_metrics = data.cuda(), target.cuda(), control_metrics.cuda()
    
        data_out = modelred(data)

        X.append(data_out.detach().cpu().numpy())
        y.append(target.detach().cpu().numpy())
        z.append(control_metrics.detach().cpu().numpy())


    X = np.vstack(X).squeeze()
    y = np.vstack(y)
    z = np.vstack(z)

    return X, y, z


# function to evaluate a set of trained classifier using AUC metric
# 'models' contains classifiers in order of binary variables to be predicted -- which are contaiend in Y
# X is a matrix of covariates
def analytics_lin(models, X, Y):

    acc = {}
    auc = {}

    for i in tqdm(range(Y.shape[1])):

        y_true = Y[:,i]
        mod = models[i]

        y_pred = np.argmax(mod.predict_proba(X),axis=1)

        # auc
        y_prob = mod.predict_proba(X)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
        auc[q_list[i]] = metrics.auc(fpr, tpr)

    return auc


# sequentially yield coordinates for blackout in an image
def sliding_window(image_shape, stepSize, windowSize):
    # slide a window across the image
    for yc in range(0, image_shape[0], stepSize):
        for xc in range(0, image_shape[1], stepSize):
            # yield the current window
            yield (yc, yc + windowSize[1], xc, xc + windowSize[0])


# calculating decrease in AUC when blocking a particular area of an image -- over 8x8 grid placed over the image
def img_area_importance(modelred, models, svd, dat, auc_true):

    patch_importance = {}

    for (y0, y1, x0, x1) in sliding_window(image_shape=(224,224), stepSize = 28, windowSize=(28,28)):

        loader = create_dataloader(dat,rand=False)

        # X_modified_raw contains image features extracted from images with a portion of the image blocked
        X_modified_raw, Y, _ = extract_data(loader, modelred, (y0, y1, x0, x1))

        # image features reduced to 500 via svd
        X_modified = svd.transform(X_modified_raw)

        auc = analytics_lin(models, X_modified, Y)

        patch_importance_q = {}  # contains -(decrease in auc after blocking of an image)
        
        for q in q_list:
            patch_importance_q[q] = auc_true[q] - auc[q]

        patch_importance[(y0, y1, x0, x1)] = patch_importance_q # decrease in auc across all variables -- for the given blocked portion of the image

    return patch_importance



# START OF THE RUN


torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


N_EPOCHS = 20
FINETUNE = True
CUDA = torch.cuda.is_available()

batch_size=10
PATH = './data'


# analysis on face vs. bodies
CASHIER = sys.argv[1]#'ALL' #'4' # 3 # 

control_list = ['02.05','03.05','04.05','05.05','06.05','07.05','08.05','09.05','10.05', '11.05', '12.05', '13.05', 
                        'time_1', 'time_2', 'time_3', 'time_4']

if CASHIER == 'ALL':

    data = pd.read_csv(PATH+'/data_face.csv')
    RESULTS = './results_face'
    control_list = control_list + ['cashier4']

elif CASHIER == '4':

    data = pd.read_csv(PATH+'/data_face.csv')
    data = data[data['cashier4']==1]
    RESULTS = './results_face_'+CASHIER

elif CASHIER == '3':

    data = pd.read_csv(PATH+'/data_face.csv')
    data = data[data['cashier4']==0]
    RESULTS = './results_face_'+CASHIER

else:
    print('Invalid data type -- terminating')
    exit()

os.makedirs(RESULTS, exist_ok=True)


# list of variables
q_list = ['alcohol', 'vodka', 'beer', 'cola', 'ice_cream', 'banana', 'bread', 'eggs', 'chocolate', 'vegetables', 'fruits', 'over_10_item_types', 'amount_over_300']


# names for variables
q_to_full_name_dict = {
    'alcohol': 'Alcohol',
    'vodka' : 'Vodka',
    'beer' : 'Beer',
    'cola': 'Cola',
    'ice_cream' : 'Ice cream',
    'banana' : 'Bananas',
    'bread' : 'Bread',
    'eggs' : 'Eggs',
    'chocolate' : 'Chocolate',
    'vegetables' : 'Vegetables',
    'fruits' : 'Fruits',
    'over_10_item_types': 'Over 10 item types on receipt', 
    'amount_over_300': 'Receipt value over 300 UAH' # 300 hrynvia ~ US $11.5 in May 2018
    }


q_to_d_dict = {} # number of levels per variable
random_threshold = {} # random guess threshold
prop = {} # proportion of class 1 in the data (vs. 0)
for i in q_list:
    q_to_d_dict[i] = np.unique(data[i]).shape[0]
    random_threshold[i] = 1.0/q_to_d_dict[i]
    prop[i] = data[i].sum()/data.shape[0]

q_d_list = [q_to_d_dict[q] for q in q_list] # vector containing number of levels per variable -- where variables are ordered as in q_list
q_d_list_cumsum = np.cumsum(q_d_list) # cumulative sum over variable levels

# total number of levels across variables
n_outs=q_d_list_cumsum[-1]


# logistic regresssion wrapper
def logistic_regression(Xtr, Xts):
    return LogisticRegression(penalty='l2', C=0.05, random_state=0, tol=1e-6, max_iter=1e7, 
        solver='lbfgs', class_weight='balanced').fit(Xtr, Xts)


# train many regressions
def train_eval_regressions(Xtr, Ytr, Xts, Yts):
    lin_models = []
    for i in tqdm(range(len(q_list))):
        clf = logistic_regression(Xtr, Ytr[:,i])
        lin_models.append(clf)
    auc = analytics_lin(lin_models, Xts, Yts)
    return auc, lin_models


# number of unique receipts
data['cid'].unique().shape

# n observations
sum(data['cashier4'] == 1) # cashier 5 on camera
sum(data['cashier4'] == 0) # cashier 4 on camera

# n unique receipts
data['cid'][data['cashier4'] == 1].unique().shape
data['cid'][data['cashier4'] == 0].unique().shape


# TRAINING

np.random.seed(999)
torch.manual_seed(999)

# load a pretrained resnet-50 network
model = get_pretrained()

# modelred is a subset of model that outputs a vector of image features per image
modelred = torch.nn.Sequential(*list(model.children())[:-1])
modelred.eval()
if CUDA:
    modelred.cuda()


n_reps = 20 # number of repeats for 5-fold cross-valaidtion
gkf = KFold(n_splits=5)

results_auc = []
results_patch_importance = []
results_auc_control = []
results_auc_combo = []

# blocking IDs - blocks are based on time period
IDs = data['block'].unique()

for rep in tqdm(range(n_reps)):

    # shuffling every repetition to get new folds via cv procedure
    np.random.shuffle(IDs)
    data_shuffled = data.sample(frac=1.0) # shufling observations too

    for trainID, testID in tqdm(gkf.split(IDs)):

        # extracting split data
        data_train = data_shuffled[data_shuffled['block'].isin(IDs[trainID])]
        data_test = data_shuffled[data_shuffled['block'].isin(IDs[testID])]

        # calculating class weights to balance data
        class_weights = calculate_class_weights(data_train)

        # creating data loaders
        loader_train = create_dataloader(data_train,rand=False)
        if FINETUNE:
            loader_train_rand = create_dataloader(data_train,rand=True)
        loader_test = create_dataloader(data_test,rand=False)

        # finetuning model
        if FINETUNE:
            finetune_and_save(loader_train_rand, loader_test)
            model = torch.load(RESULTS+"/finetuned_model")
            modelred = torch.nn.Sequential(*list(model.children())[:-1])
            modelred.eval()
            if CUDA:
                modelred.cuda()

        # extracting image features, labels, and control variables
        X_train_raw, Y_train, Z_train = extract_data(loader_train, modelred)
        X_test_raw, Y_test, Z_test = extract_data(loader_test, modelred)

        # reducing number of features
        svd = TruncatedSVD(n_components=500, random_state=0, n_iter=100).fit(X_train_raw)
        X_train = svd.transform(X_train_raw)
        X_test = svd.transform(X_test_raw)

        # training linear models - image features only
        auc, lin_models = train_eval_regressions(X_train, Y_train, X_test, Y_test)
        results_auc.append(auc)

        # image area importance
        patch_importance = img_area_importance(modelred, lin_models, svd, data_test, auc)
        results_patch_importance.append(patch_importance)

        # control variables
        auc, lin_models = train_eval_regressions(Z_train, Y_train, Z_test, Y_test)
        results_auc_control.append(auc)

        # image features + control variables
        auc, lin_models = train_eval_regressions(np.concatenate([X_train, Z_train],1), Y_train, np.concatenate([X_test, Z_test],1), Y_test)
        results_auc_combo.append(auc)   


# saving results of the run
pd.DataFrame(results_auc).to_csv(RESULTS+'/crossvalidation_auc.csv', index=False)
pd.DataFrame(results_auc_control).to_csv(RESULTS+'/crossvalidation_auc_control.csv', index=False)
pd.DataFrame(results_auc_combo).to_csv(RESULTS+'/crossvalidation_auc_combo.csv', index=False)


# saving patch_importance
patch_importance = {}
for q in q_list:

    arr = np.zeros((224,224))
    
    for (y0, y1, x0, x1) in sliding_window(image_shape=(224,224), stepSize = 28, windowSize=(28,28)):
        arr[y0:y1, x0:x1] = np.mean([i[(y0, y1, x0, x1)][q] for i in results_patch_importance])

    patch_importance[q] = arr.tolist()


with open(RESULTS+'/patch_importance.json', 'w') as fjson:
    json.dump(patch_importance, fjson)







# VISUALIZATIONS
colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe', 
    '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', 
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']


# extracting auc data for each fold of crossvalidation (cv) and each variable
results_auc = pd.read_csv(RESULTS+'/crossvalidation_auc.csv')
results_auc = results_auc.stack().reset_index()
results_auc.columns = ['cv_fold', 'var_name', 'auc']

# calculating mean AUC mean and sd across cv folds for each variable
results_auc = results_auc[['var_name', 'auc']].groupby(['var_name'],sort=False).agg(['mean','std']).reset_index()
results_auc.columns = results_auc.columns.map('_'.join).str.strip('_')

# encoding full name
results_auc['var_name_full'] = [q_to_full_name_dict[q] for q in results_auc['var_name'].tolist()]

# calculating confidence interval on auc for each variables
results_auc['auc_l'] = results_auc['auc_mean'] - 2*results_auc['auc_std']
results_auc['auc_u'] = results_auc['auc_mean'] + 2*results_auc['auc_std']

# mean value of the variable in the full data
temp = data[q_list].mean().reset_index()
temp.columns = ['index', 'var_mean']
results_auc = results_auc.merge(temp, left_on='var_name', right_on='index')
results_auc = results_auc.drop('index',1)

# p values
results_auc['p_val'] = [scipy.stats.norm(results_auc['auc_mean'].iloc[i], results_auc['auc_std'].iloc[i]).cdf(0.5) for i in range(results_auc.shape[0])]

# save auc analysis
results_auc.to_csv(RESULTS+'/results_auc.csv')



# INDIVIDUAL VARIABLE MEANS
results_auc = results_auc.sort_values('p_val', ascending=True)
results_auc_filtered = results_auc#[results_auc['auc_l']>0.5]

# % variables with significant AUC 
results_auc_filtered.shape[0]/results_auc.shape[0]






# CORRELATION MATRIX AND FACTOR ANALYSIS


# import seaborn as sns
df = data[q_list].copy()

# correlation matrix
Xcorr = df.corr().values

# distances based on sign-less correlation matrix
d = sch.distance.squareform(1-np.abs(Xcorr))

# hierarchical clustering linkage
L = sch.linkage(d, method='single')

sns_plot = sns.clustermap(Xcorr, figsize=(10, 10), row_linkage=L, col_linkage=L, xticklabels=1, yticklabels=1, annot=True, annot_kws={"size": 10}, fmt='.2f')
plt.setp(sns_plot.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(sns_plot.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

ax = sns_plot.ax_heatmap

cols = [df.columns[i] for i in list(sns_plot.data2d.columns)]

ax.set_xticklabels([q_to_full_name_dict[i] for i in cols], fontsize = 20) #ax.get_xmajorticklabels()
ax.set_yticklabels([q_to_full_name_dict[i] for i in cols], fontsize = 20)

# ax.set_xticklabels(list(range(0,len(cols))), fontsize = 20) #ax.get_xmajorticklabels()
# ax.set_yticklabels(list(range(0,len(cols))), fontsize = 20)

sns_plot.fig.axes[-1].tick_params(labelsize=15)

sns_plot.savefig(RESULTS+'/var_corr1.pdf')
plt.close()

pd.DataFrame.from_dict({'Variable':[q_to_full_name_dict[i] for i in cols]}).reset_index().to_csv(RESULTS+'/var_corr1_order.csv',index=False)


# calculating mean and sd across cv folds for each variable
temp = df[cols].stack().reset_index()
temp.columns = ['respondent', 'var_name', 'value']
temp['var_name_full'] = [q_to_full_name_dict[q] for q in temp['var_name'].tolist()]
temp = temp[['var_name_full', 'var_name', 'value']].groupby(['var_name_full', 'var_name'],sort=False).agg(['mean','std']).reset_index()
temp.to_csv(RESULTS+'/var_corr1_order_summary.csv')



# INDIVIDUAL VARIABLES

# Func to draw line segment
def newline(p1, p2, linewidth =1.0, color='firebrick'):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], linewidth = linewidth, color=color)
    ax.add_line(l)
    return l

# plot group results as group chart with error bars
plt.figure(figsize=(6,6), dpi=300)

# sets vertical index
plt.hlines(y=results_auc_filtered['var_name_full'].tolist(), xmin=0, xmax=1, color='gray', alpha=0.0, linewidth=.5, linestyles='dashdot')

# plots dots
plt.scatter(results_auc_filtered['auc_mean'].values, results_auc_filtered['var_name_full'].tolist()[:42], marker='o', s = 75., color='firebrick')

# line segments
for i, p1, p2 in zip(results_auc_filtered['var_name_full'], 
    results_auc_filtered['auc_l'].values, 
    results_auc_filtered['auc_u'].values):
    newline([p1, i], [p2, i])

plt.axvline(x=0.5, color='k', linestyle=':')

plt.xlim([0.4,1])
plt.xlabel('AUC')
plt.gca().invert_yaxis()

red_patch = plt.plot([],[], marker="o", ms=10, ls="", mec=None, color='firebrick', label="AUC")
red_line = mlines.Line2D([0], [0], linewidth = 1.0, color='firebrick', label="[AUC-2SE : AUC+2SE]")
leg = plt.legend(handles=[red_patch[0], red_line], loc='lower right', bbox_to_anchor=(1., -0.2), ncol=2, fontsize=11.)

plt.gca().spines["top"].set_visible(False)    
plt.gca().spines["bottom"].set_visible(False)    
plt.gca().spines["right"].set_visible(False)    
plt.gca().spines["left"].set_visible(False)   

plt.grid(axis='both', alpha=.4, linewidth=.1)

plt.savefig(RESULTS+'/variable_auc.pdf', bbox_inches='tight', transparent=True)
plt.close()



# number of significantly predictable variables by model
def multi_auc(paths, model_names, saved):

    colind = [0,9,14]

    res = []

    for p in paths:

        temp = pd.read_csv(p)
        temp = temp.stack().reset_index()
        temp.columns = ['cv_fold', 'var_name', 'auc']

        # calculating mean AUC mean and sd across cv folds for each variable
        temp = temp[['var_name', 'auc']].groupby(['var_name'],sort=False).agg(['mean','std']).reset_index()
        temp.columns = temp.columns.map('_'.join).str.strip('_')

        # calculating confidence interval on auc for each variables
        temp['auc_l'] = temp['auc_mean'] - 2*temp['auc_std']
        temp['auc_u'] = temp['auc_mean'] + 2*temp['auc_std']

        temp['p_val'] = [scipy.stats.norm(temp['auc_mean'].iloc[i], temp['auc_std'].iloc[i]).cdf(0.5) for i in range(temp.shape[0])]

        temp = temp.sort_values('p_val', ascending=True)

        temp['var_name_full'] = [q_to_full_name_dict[q] for q in temp['var_name'].tolist()]

        res.append(temp)

    for i in range(len(model_names)):
        res[i].to_csv(RESULTS+'/results ' + model_names[i] + '.csv')


    # plotting
    plt.figure(figsize=(6,6), dpi=300)

    ind = np.arange(res[0]['var_name'].shape[0])

    # sets vertical index
    plt.hlines(y=res[0]['var_name_full'].tolist(), xmin=0, xmax=1, color='gray', alpha=0.0, linewidth=.5, linestyles='dashdot')

    for i in range(len(model_names)):

        # ordering variables as the order in the first evaluated model
        temp = res[i].set_index('var_name')
        temp = temp.loc[res[0]['var_name']]

        # index used for plotting
        ind_adj = ind + (i-1)*0.25

        # plots dots
        plt.scatter(res[i]['auc_mean'].values, ind_adj, marker='o', s = 75., color=colors[colind[i]])

        # line segments
        for j, p1, p2 in zip(ind_adj, 
            res[i]['auc_l'].values, 
            res[i]['auc_u'].values):
            newline([p1, j], [p2, j], color=colors[colind[i]])

    plt.axvline(x=0.5, color='k', linestyle=':')

    plt.xlim([0.2,1])
    plt.xlabel('AUC')
    plt.gca().invert_yaxis()

    red_patch = plt.plot([],[], marker="o", ms=10, ls="", mec=None, color='k', label="AUC")
    red_line = mlines.Line2D([0], [0], linewidth = 1.0, color='k', label="[AUC-2SE : AUC+2SE]")
    cpatch1 = plt.plot([],[], marker=".", ms=10, ls="", mec=None, color=colors[colind[0]], label=model_names[0])
    cpatch2 = plt.plot([],[], marker=".", ms=10, ls="", mec=None, color=colors[colind[1]], label=model_names[1])
    cpatch3 = plt.plot([],[], marker=".", ms=10, ls="", mec=None, color=colors[colind[2]], label=model_names[2])
    leg = plt.legend(handles=[red_patch[0], cpatch1[0], cpatch3[0], red_line, cpatch2[0]], loc='lower right', bbox_to_anchor=(1., -0.3), ncol=2, fontsize=11.)


    plt.savefig(saved+'.pdf', bbox_inches='tight', transparent=True)
    plt.close()


paths = [
    RESULTS+'/crossvalidation_auc.csv',
    RESULTS+'/crossvalidation_auc_control.csv',
    RESULTS+'/crossvalidation_auc_combo.csv'
    ]


model_names = [
    'Deep image features',
    'Control variables (time of day, day)',
    'Deep image features + Controls'
    ]

multi_auc(paths, model_names, RESULTS+'/multi_auc')



# number of significantly predictable variables by model
def waterfall(paths, model_names, saved):

    res = []

    for p in paths:

        temp = pd.read_csv(p)
        temp = temp.stack().reset_index()
        temp.columns = ['cv_fold', 'var_name', 'auc']

        # calculating mean AUC mean and sd across cv folds for each variable
        temp = temp[['var_name', 'auc']].groupby(['var_name'],sort=False).agg(['mean','std']).reset_index()
        temp.columns = temp.columns.map('_'.join).str.strip('_')

        # calculating confidence interval on auc for each variables
        temp['auc_l'] = temp['auc_mean'] - 2*temp['auc_std']
        temp['auc_u'] = temp['auc_mean'] + 2*temp['auc_std']

        temp['p_val'] = [scipy.stats.norm(temp['auc_mean'].iloc[i], temp['auc_std'].iloc[i]).cdf(0.5) for i in range(temp.shape[0])]

        temp['var_name_full'] = [q_to_full_name_dict[q] for q in temp['var_name'].tolist()]

        temp = temp.sort_values('p_val', ascending=True)

        res.append(temp)


    predictable_n = []
    predictable_n_fdr = []

    for i in range(len(res)):

        # number of predictable variables by 2 se criterion
        t = res[i]['auc_l']
        predictable_n.append(((t/0.5)>1).sum())

        # number of predictable variables by fdr control criterion
        alpha = 0.05 # desired control level for FDR
        slope = alpha/res[i].shape[0]

        below = res[i]['p_val'].values <= slope * np.array(list(range(1,1+res[i]['p_val'].shape[0])))
        if sum(below) > 0:
            tot_fdr = np.max(np.where(below)[0])+1
        else:
            tot_fdr = 0

        predictable_n_fdr.append(tot_fdr)

    predictable_n_fdr = np.array(predictable_n_fdr)
    predictable_n = np.array(predictable_n)

    # plotting
    plt.figure(figsize=(6,6), dpi=300)

    plt.plot(predictable_n, model_names, '-o', color=colors[0], label='2SE significance')
    plt.plot(predictable_n_fdr, model_names, '--D', color=colors[9], label='BH(0.05) significance')


    plt.xlabel('Number of predictable variables')
    plt.gca().invert_yaxis()

    plt.gca().spines["top"].set_visible(False)    
    plt.gca().spines["bottom"].set_visible(False)    
    plt.gca().spines["right"].set_visible(False)    
    plt.gca().spines["left"].set_visible(False)   

    plt.grid(axis='both', alpha=.4, linewidth=.1)

    plt.legend()

    plt.savefig(saved+'.pdf', bbox_inches='tight', transparent=True)
    plt.close()

    pd.DataFrame([model_names,predictable_n.tolist(),predictable_n_fdr.tolist()]).to_csv(saved+'.csv',index=False)


paths = [
    RESULTS+'/crossvalidation_auc.csv',
    RESULTS+'/crossvalidation_auc_control.csv',
    RESULTS+'/crossvalidation_auc_combo.csv'
    ]


model_names = [
    'Deep image features',
    'Control variables (time of day, day)',
    'Deep image features + Controls'
    ]

waterfall(paths, model_names, RESULTS+'/waterfall')




# extracting raw images
def extract_raw_images(loader):

    images = []
    for batch_i, var in tqdm(enumerate(loader)):

        image_batch, _, _ = var
        images.append(image_batch.detach().cpu().numpy())

    images = np.vstack(images).squeeze()
    return images


loader_full = create_dataloader(data, rand=False)
raw_images = extract_raw_images(loader_full)

raw_images= (raw_images - raw_images.min())/ (raw_images.max()-raw_images.min())


# across all images
mean_image = np.transpose(raw_images.mean(0), (1, 2, 0))
mean_image = Image.fromarray(np.uint8(mean_image*255.0))
mean_image.save(RESULTS+'/mean_image.png')


# creating square tiled image
n = 8
h = 224

# tile
fig = np.zeros(shape=(h*n,h*n,3))
for i in range(n):
    for j in range(n):
        ind = n*i+j
        img = np.transpose(raw_images[ind], (1, 2, 0))
        fig[(i*h):((i+1)*h),(j*h):((j+1)*h)] = img.copy()

tiled_image = Image.fromarray(np.uint8(fig*255.0))
tiled_image.save(RESULTS+'/tiled_image.png')


# VISUALIZING IMAGE AREA IMPORTANCE

background_image = np.transpose(raw_images.mean(0), (1, 2, 0))
background_image = background_image*255.0
background_image = np.dstack((background_image,(np.zeros(shape=(224,224,1))+255)))
background_image = Image.fromarray(np.uint8(background_image))

# create directories to store area importance images
os.makedirs(RESULTS+'/img_imp', exist_ok=True)
os.makedirs(RESULTS+'/img_imp_background', exist_ok=True)

# path importance loading
patch_importance = json.loads(open(RESULTS+'/patch_importance.json').read())

for q in q_list:
    arr = np.array(patch_importance[q])
    arr = (arr - arr.min())/(arr.max()-arr.min())

    im = Image.fromarray(np.uint8(plt.cm.get_cmap('YlOrRd')(arr)*255.0))
    im.save(RESULTS+'/img_imp/'+q+'.png')

    im = np.uint8((plt.cm.get_cmap('YlOrRd')(arr))*255.0)
    im[:,:,3] = 96
    im = Image.fromarray(im)
    im = Image.alpha_composite(background_image, im)
    im.save(RESULTS+'/img_imp_background/'+q+'.png')


