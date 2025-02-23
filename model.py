import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Normal
from layers import GraphConvolution
import pandas as pd

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.norm= torch.nn.BatchNorm1d(nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x=self.gc1(x, adj)
        x=self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GCN_VAE(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_VAE, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
    def forward(self, x, adj):
        mu = F.relu(self.gc1(x, adj))
        mu= F.dropout(mu, self.dropout, training=self.training)
        var = F.relu(self.gc2(x, adj))
        var= F.dropout(var, self.dropout, training=self.training)
        var = torch.exp(var).sqrt()
        z = Normal(mu, var).rsample()
        return z,mu,var

class VGAE(nn.Module):
    def __init__(self,nfeat,nhid,out,dropout):
        super(VGAE, self).__init__()
        self.GCN=GCN(nfeat,nhid,dropout)
        self.GCN_VAE=GCN_VAE(nhid,out,dropout)
        self.dropout = dropout
    def forward(self,x,adj):
        H1= self.GCN(x, adj)
        z,mu, var= self.GCN_VAE(H1, adj)
        return mu,var,z


class decoder(torch.nn.Module):
    def __init__(self, nfeat,nhid,out,dropout):
        super(decoder, self).__init__()
        self.decoder1 = torch.nn.Sequential(
            torch.nn.Linear(out, nhid),
            torch.nn.BatchNorm1d(nhid),
            torch.nn.ReLU()
        )
        self.decoder4 = torch.nn.Linear(nhid, nfeat)
        self.dropout = dropout
    def forward(self, emb):
        x = self.decoder1(emb)
        x=F.dropout(x, self.dropout, training=self.training)
        x = self.decoder4(x)
        return x




def poe(mus, vars,mask):
    """
    Product of Experts
    - mus: [mu_1, ..., mu_M], where mu_m is N * K
    - logvars: [logvar_1, ..., logvar_M], where logvar_m is N * K
    """
    #print(mus,'mus')
    #print(vars,'vars')
    T=torch.reciprocal(vars)*mask
    T_sum=T.sum(1)+1
    #print(T_sum)
    pd_mu=(mus*T).sum(1)/T_sum
    pd_var=1/T_sum
    #print(pd_mu,'pd_mu')
    #print(pd_var,'pd_var')
    return pd_mu, pd_var




















class multi_all(torch.nn.Module):
    def __init__(self, nfeat1,nfeat2,nhid,out,dropout):
        super(multi_all, self).__init__()
        self.encoder_rna = torch.nn.Sequential(
            torch.nn.Linear(nfeat1, nhid),
            nn.BatchNorm1d(nhid),
            torch.nn.ReLU()
        )
        self.encoder_rna_mu= torch.nn.Linear(nhid, out)
        self.encoder_rna_var= torch.nn.Linear(nhid, out)

        self.encoder_atac = torch.nn.Sequential(
            torch.nn.Linear(nfeat2, nhid),
            nn.BatchNorm1d(nhid),
            torch.nn.ReLU()
        )
        self.encoder_atac_mu= torch.nn.Linear(nhid, out)
        self.encoder_atac_var= torch.nn.Linear(nhid, out)
        self.dropout = dropout

        self.decoder = decoder(nfeat1+nfeat2, nhid, out, dropout)
        self.cluster_layer = nn.Parameter(torch.Tensor(19, out), requires_grad=True)
    def forward(self,rna,atac,index_rna,index_atac):
        index_rna=index_rna.int()
        index_atac=index_atac.int()
        h_rna=F.dropout(self.encoder_rna(rna))
        zm_rna=self.encoder_rna_mu(h_rna)
        zv_rna=torch.exp(self.encoder_rna_var(h_rna))
        h_atac=F.dropout(self.encoder_atac(atac))
        zm_atac=self.encoder_atac_mu(h_atac)
        zv_atac=torch.exp(self.encoder_atac_var(h_atac))
        total_number=max(max(index_atac),max(index_rna))
        #print(zm_rna,'zm_rna')
        mu=torch.ones([len(rna),2,16])
        var=torch.ones([len(rna),2,16])
        mask=torch.zeros([len(rna) ,2,16])
        # mu=torch.ones([len(rna),2,16]).cuda()
        # var=torch.ones([len(rna),2,16]).cuda()
        # mask=torch.zeros([len(rna) ,2,16]).cuda()
        for i in range(len(zm_rna)):
            mu[index_rna[i]][0]=zm_rna[i]
            var[index_rna[i]][0] = zv_rna[i]
            mask[index_rna[i]][0] = torch.ones(16)
        for j in range(len(zm_atac)):
            mu[index_atac[j]][1] = zm_atac[j]
            var[index_atac[j]][1] = zv_atac[j]
            mask[index_atac[j]][1] = torch.ones(16)
        #print(mu.is_cuda,zm_rna.is_cuda)
        #print(mask)
        z_mu,  z_var=poe(mu,var,mask)
        #print(z_mu.size(),'size1')
        #print(z_var.size(),'size2')
        z = Normal(z_mu, z_var).rsample()
        q = 1.0 / (1.0 + torch.sum(torch.pow((z).unsqueeze(1) - self.cluster_layer, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        #print(z)
        recon_x = self.decoder(z)
        return recon_x,z_mu, z_var,z,q






