from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from model import multi_all
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from utils import features_construct_graph
from fusion_utils import *
import scanpy as sc
from torch.distributions import Normal, kl_divergence
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, calinski_harabasz_score
import anndata as ad
from sklearn.metrics.cluster import contingency_matrix

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def generate_triplets(label, generate):
    anchor = []
    positive = []
    negative = []
    num = 1

    while num > 0:
        idx1 = random.sample(range(label.shape[0]), 2)
        if label[idx1[0]] == label[idx1[1]]:
            idx2 = random.sample(range(label.shape[0]), 1)
            if label[idx1[0]] != label[idx2]:
                anchor.append(idx1[0])
                positive.append(idx1[1])
                negative.append(idx2)
        if len(anchor) == generate:
            break
    return np.array(anchor), np.array(positive), np.array(negative).flatten()


def tripletloss(anchor, positive, negative, margin_constant):
    # loss = max(d(anchor, negative) - d(anchor, positve) + margin, 0), margin > 0
    # d(x, y) = q(x) * q(y)
    negative_dis = torch.sum(anchor * negative, dim=1)
    positive_dis = torch.sum(anchor * positive, dim=1)
    margin = margin_constant * torch.ones(negative_dis.shape).cuda()
    diff_dis = negative_dis - positive_dis
    penalty = diff_dis + margin
    triplet_loss = 1 * torch.max(penalty, torch.zeros(negative_dis.shape).cuda())
    return torch.mean(triplet_loss)

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
    index_rna=[x for x in range(len(rna))]
    print(rna)
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=4000)
    rna = rna[:, rna.var.highly_variable]
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    fts = torch.tensor(rna.X.todense())
    print('rna fts shape:', fts.shape)
    return fts,label,index_rna


# load ATAC modality data
def load_atac_data_h5(data_dir):
    atac = sc.read_h5ad(data_dir)
    index_atac=[x for x in range(len(atac))]
    sc.pp.highly_variable_genes(atac, flavor='seurat_v3', n_top_genes=20000)
    atac = atac[:, atac.var.highly_variable]
    c = atac.X.todense()
    atac.X = np.array(idf(torch.tensor(c)))
    sc.pp.scale(atac)

    fts = torch.tensor(atac.X)
    print('atac fts shape:', fts.shape)
    return fts,index_atac


def load_data(data_dir_rna,
              data_dir_atac,name):
    print("load data:", data_dir_rna)
    ft_rna,lbls,index_rna = load_rna_data_h5(data_dir_rna)
    rna = ft_rna.float()
    lbls = lbls
    print(ft_rna.shape)
    print("load data:", data_dir_atac)
    ft_atac,index_atac = load_atac_data_h5(data_dir_atac)
    atac = ft_atac.float()
    print(ft_atac.shape)

    index_rna = torch.FloatTensor(index_rna)
    index_atac = torch.FloatTensor(index_atac)


    rna_coo_index = np.load('./mosaic_data/datasets/'+name+'/rnaadj_new.npz')
    rna_coo = csr_matrix((rna_coo_index['data'], (rna_coo_index['row'], rna_coo_index['col'])),
                         shape=rna_coo_index['shape']).tocoo()

    atac_coo_index = np.load('./mosaic_data/datasets/'+name+'/atacadj_new.npz')
    atac_coo = csr_matrix((atac_coo_index['data'], (atac_coo_index['row'], atac_coo_index['col'])),
                          shape=atac_coo_index['shape']).tocoo()



    print("done")

    return rna, atac, lbls, rna_coo, atac_coo, index_rna, index_atac


def train():
    model.train()
    optimizer.zero_grad()
    recon_x, mu, var, z, q = model(rna, atac, index_rna, index_atac)
    norm_z = torch.nn.functional.normalize(z, dim=1)
    # triplet_loss
    anchor, positive, negative = generate_triplets(label, len(rna))
    triplet_loss = 10 * tripletloss(z[anchor], z[positive], z[negative], 0.1) * len(rna)

    # cluster_loss
    p = target_distribution(q.data)
    cluster_loss = torch.nn.functional.kl_div(q.log(), p, reduction='batchmean')

    # print(x.size(0), '11111111111111')
    loss_fuction = nn.MSELoss()
    # print( loss_fuction(x,recon_x),'22222222222222')
    index_rna111 = np.array(index_rna.cpu())
    index_atac111 = np.array(index_atac.cpu())


    reconstruction_loss1 = loss_fuction(rna, recon_x[index_rna111, :4000]) * rna.size(-1)
    reconstruction_loss2 = loss_fuction(atac, recon_x[index_atac111, 4000:]) * atac.size(-1)

    # noise_x=x-recon_x
    # noise_y=y-recon_y
    # var_x=
    recon_loss = reconstruction_loss1 + reconstruction_loss2
    kl_loss = 10 * kl_div_sqrt(mu, var)

    reg_loss_1 = 100 * graphloss(norm_z, norm_z, rna_coo)
    reg_loss_2 = 100 * graphloss(norm_z, norm_z, atac_coo)

    reg_loss = reg_loss_1 + reg_loss_2

    contra_loss_1 = 100 * contrastive_loss(norm_z, norm_z, rna_coo, 0.5)
    contra_loss_2 = 100 * contrastive_loss(norm_z, norm_z, atac_coo, 0.5)


    contra_loss = contra_loss_1 + contra_loss_2

    total_loss = config.alpha * recon_loss + config.gamma * kl_loss - reg_loss + triplet_loss + contra_loss

    emb = pd.DataFrame(z.cpu().detach().numpy()).fillna(0).values

    mu = pd.DataFrame(mu.cpu().detach().numpy()).fillna(0).values
    var = pd.DataFrame(var.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mu, var, recon_loss, kl_loss, reg_loss, triplet_loss, cluster_loss, contra_loss, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['10X_PBMC','Ma_2020','Chen_2019','Xie_2023']
    for i in range(len(datasets)):
        datasetname = datasets[i]
        config_file = './config/config_mosaic.ini'

        data_dir_rna = './test/datasets/' + datasetname + '/RNA_pre.h5ad'
        data_dir_atac = './test/datasets/' + datasetname + '/ATAC_pre.h5ad'


        rna, atac,  label, rna_coo, atac_coo, index_rna, index_atac = load_data(
            data_dir_rna,data_dir_atac,datasetname)

        savepath = './mosaic_data/datasets/' + datasetname

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        number_cluster = sc.read_h5ad('./test/datasets/' + datasetname + '/RNA_pre.h5ad')
        n_cluster = len(np.unique(number_cluster.obs['cell_type']))

        config.epochs = 200
        config.epochs = config.epochs + 1

        if cuda:
            rna = rna.cuda()

            atac = atac.cuda()

            index_rna = index_rna.cuda()
            index_atac = index_atac.cuda()

        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        print(datasetname, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)

        model = multi_all(nfeat1=4000, nfeat2=20000, nhid=config.nhid1,
                            out=16,
                            dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        epoch_max = 0
        ari_max = 0
        acc_max = 0
        nmi_max = 0
        ASW_max = 0
        PS_max = 0
        HS_max = 0
        mu_max = []
        var_max = []
        emb_max = []
        for epoch in range(config.epochs):
            emb, mu, var, recon_loss, kl_loss, reg_loss, triplet_loss, cluster_loss, contra_loss, total_loss, = train()
            print(datasetname, ' epoch: ', epoch, ' recon_loss = {:.2f}'.format(recon_loss),
              ' kl_loss = {:.2f}'.format(kl_loss), ' reg_loss = {:.2f}'.format(reg_loss),
              ' triplet_loss = {:.2f}'.format(triplet_loss), ' cluster_loss = {:.2f}'.format(cluster_loss),
              ' contra_loss = {:.2f}'.format(contra_loss),
              ' total_loss = {:.2f}'.format(total_loss))
            embedding_name = []
            cell_name = []
            for s in range(emb.shape[1]):
                embedding_name.append(str(s))

            for k in range(emb.shape[0]):
                cell_name.append(str(k))
            embedding_name = pd.DataFrame(index=embedding_name)
            cell_name = pd.DataFrame(index=cell_name)
            adata_learned = ad.AnnData(emb, obs=cell_name, var=embedding_name)
            adata_learned.obs['cell_type'] = label
            sc.pp.neighbors(adata_learned, use_rep='X', n_neighbors=30)
            sc.tl.leiden(adata_learned, resolution=1)  # best
            idx = adata_learned.obs['leiden']

            ari = metrics.adjusted_rand_score(label, idx)
            nmi = metrics.normalized_mutual_info_score(label, idx)
            # ASW=(np.round(silhouette_score(emb,idx),3))
            ASW = 0
            PS = (np.round(purity_score(label, idx), 3))
            HS = (np.round(homogeneity_score(label, idx), 3))
            if ari > ari_max:
                ari_max = ari
                nmi_max = nmi
                ASW_max = ASW
                PS_max = PS
                HS_max = HS
                # acc_max=acc
                epoch_max = epoch
                emb_max = emb
            print('NMI：', nmi_max)
            print('ARI：', ari_max)
            print('ASW：', ASW_max)
            print('PS：', PS_max)
            print('HS：', HS_max)
        print('NMI：', nmi_max)
        print('ARI：', ari_max)

        pd.DataFrame(emb_max).to_csv(savepath + '/' + datasetname + '_vae_emb.csv')
