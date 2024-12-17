from __future__ import division
from __future__ import print_function

import numpy as np
import torch.optim as optim

import os
import argparse

from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from  fusion_utils import *
from utils import *
import scanpy as sc
from torch.distributions import Normal, kl_divergence
import anndata as ad
from sklearn.metrics.cluster import contingency_matrix
from scipy.sparse import save_npz

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def g_mean(x):
    sum=torch.exp(torch.sum(torch.log(x))/len(x))
    return sum.unsqueeze_(0)

def cls(data):
    a = []
    for i in range(len(data)):
        index = np.where(data[i] != 0)
        gmeans = g_mean(data[i][index])
        a.append(gmeans)
    gmean = torch.cat(a, dim=0)
    X = (data.T / gmean).T
    mask = np.where(X == 0)
    X_log = torch.log(X)
    X_log[mask] = 0
    return X_log

def idf(X):
    freq = torch.tensor(np.where(X == 0, 0, 1))
    frequency = torch.sum(freq, dim=0) + 1
    count = X.shape[0]
    idf = torch.log(torch.tensor(count, dtype=torch.float) / frequency)
    return X * idf


def load_rna_data_h5(data_dir):
    rna = sc.read_h5ad(data_dir)
    label=np.array(rna.obs['cell_type'])

    print(rna)
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=4000)
    rna = rna[:, rna.var.highly_variable]
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    fts = torch.tensor(rna.X.todense())
    print('rna fts shape:', fts.shape)
    return fts,label


# load ATAC modality data
def load_atac_data_h5(data_dir):
    atac = sc.read_h5ad(data_dir)

    sc.pp.highly_variable_genes(atac, flavor='seurat_v3', n_top_genes=20000)
    atac = atac[:, atac.var.highly_variable]
    c = atac.X.todense()
    atac.X = np.array(idf(torch.tensor(c)))
    sc.pp.scale(atac)

    fts = torch.tensor(atac.X)
    print('atac fts shape:', fts.shape)
    return fts



def load_data(data_dir_rna,
              data_dir_atac):
    print("load data:", data_dir_rna)
    ft_rna,lbls = load_rna_data_h5(data_dir_rna)
    rna = ft_rna.float()
    lbls = lbls
    print(ft_rna.shape)
    print("load data:", data_dir_atac)
    ft_atac = load_atac_data_h5(data_dir_atac)
    atac = ft_atac.float()
    print(ft_atac.shape)


    fadj1_11 = batch_graph(ft_rna, ft_rna, k_numeber=31)

    fadj2_11 = batch_graph(ft_atac, ft_atac, k_numeber=31)

    return rna, atac, lbls, fadj1_11,fadj2_11


datasets = ['10K_PBMC','10X_PBMC','Ma_2020','Chen_2019','Xie_2023']
for i in range(len(datasets)):
    datasetname=datasets[i]

    data_dir_rna = './test/datasets/'+datasetname+'/RNA_pre.h5ad'
    data_dir_atac = './test/datasets/'+datasetname+'/ATAC_pre.h5ad'

    rna,atac, label,rna_adj,atac_adj = load_data(data_dir_rna,data_dir_atac)
    print(rna_adj)

    save_npz( './mosaic_data/datasets/'+datasetname+'/rnaadj_new.npz', rna_adj)
    save_npz('./mosaic_data/datasets/'+datasetname+'/atacadj_new.npz', atac_adj)
















