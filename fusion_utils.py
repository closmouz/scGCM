import scipy.sparse as sp
import sklearn
import torch
import networkx as nx
from sklearn.cluster import KMeans
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.cluster import contingency_matrix

EPS = 1e-15


# def normalize(adata, highly_genes=3000, normalize_input=True):
#     print("start select HVGs")
#     adata = np.delete(adata, np.mean(adata, axis=0) < 0.04, axis=1)
#     adata1 = np.int64(adata > 0)
#     adata = np.delete(adata, np.sum(adata1, axis=0) < 300, axis=1)
#     adata = np.delete(adata, np.std(adata, axis=0) < 0.1, axis=1)
#     top_k_idx = np.std(adata, axis=0).argsort()[::-1][0:highly_genes]
#     top_k_idx = top_k_idx[::-1]
#     adata = adata[:, top_k_idx]
#     if normalize_input:
#         adata = adata / np.sum(adata, axis=1).reshape(-1, 1) * 10000
#     return sc.AnnData(adata)

def graphloss(data1, data2, adj):
    """

    :param data: 归一化后的data
    :param adj:  coo图
    :return:
    """
    loss = 0
    for i in range(adj.nnz):
        loss = loss + torch.log(torch.sigmoid(torch.matmul(data1[adj.row[i]], data2[adj.col[i]])))
    #print(loss)
    #print(len(data1))
    #print(loss/len(data1))
    return loss/len(data1)


def contrastive_loss(data1, data2, adj, tau):
    idx1 = []
    idx2 = []
    sample_idx = np.random.choice(np.arange(len(data1)), 6000)

    loss = 0
    for i in range(6000):
        index = np.where(adj.row == sample_idx[i])
        if len(list(index[0]))>0:
            #print(index[0])
            idx = random.sample(list(index[0]), 1)
            #print(idx)
            idx1.append(sample_idx[i])
            idx2.append(adj.col[idx][0])
    #print(idx1)
    data = data1[idx1]
    #print(data.size())
    #print(idx2)
    #print(data2[idx2].size())
    #print(idx2)
    loss = 0
    #print(data1)
    f = lambda z: torch.exp(z / tau)
    X12=f(torch.matmul(data,data2[idx2].T))

    X1=f(torch.matmul(data,data.T))
    X2=f(torch.matmul(data2[idx2],data2[idx2].T))
    for j in range(len(data)):
        #f = lambda z: torch.exp(z / tau)
        #print(data1,data2[idx2])
        #X12=f(torch.matmul(data1,data2[idx2].T))
        #X1=f(torch.matmul(data1,data.T))
        #X2=f(torch.matmul(data2[idx2],data2[idx2].T))

        loss1=torch.log(X12[j,j]/(torch.sum(X1[j:])+torch.sum(X12[j:])-X1[j,j]))
        loss2=torch.log(X12[j,j]/(torch.sum(X2[j:])+torch.sum(X12[j:])-X2[j,j]))
        #print(loss1,'loss1')
        #print(loss2,'loss2')
        '''
        loss1 = -torch.log(f(torch.mm(data[j], data[j].T)) / (
                    torch.sum(f(torch.mm(data[j], data2[idx2].T))) + torch.sum(f(torch.mm(data[j], data.T))) - f(
                torch.mm(data[j], data[j].T))))

        loss2 =-torch.log(f(torch.mm(data2[idx2[j]], data2[idx2[j]].T)) / (
                    torch.sum(f(torch.mm(data2[idx2[j]], data2[idx2].T))) + torch.sum(f(torch.mm(data2[idx2[j]], data.T))) - f(
                torch.mm(data2[idx2[j]], data2[idx2[j]].T))))
        '''
        loss=loss+(loss1+loss2)/12000
    return loss

def coo_adj(data1, data2, k_numeber):
    row = []
    col = []
    values = []
    for i in range(len(data1)):
        distances = torch.pairwise_distance(data1[i], data2)
        nn_index = np.array(torch.argsort(distances))
        row = np.concatenate((row, np.ones(k_numeber) * i))
        col = np.concatenate((col, nn_index[:k_numeber]))
        values = np.concatenate((values, np.ones(k_numeber)))
    #print(row)
    #print(col)
    #print(values)
    coo = coo_matrix((values, (row, col)), shape=(len(data1), len(data2)), dtype=np.float64)
    
    return coo


def coo_pairs(adj1, adj2):
    X1 = torch.tensor(adj1.todense())
    X2 = torch.tensor(adj2.todense())
    X = X1 - X1.multiply(X1 > X2.T)
    nozero = np.array(X)
    row, col = np.nonzero(nozero)
    values = nozero[row, col]
    csr = csr_matrix((values, (row, col)), shape=(len(X1), len(X2)))
    return csr.tocoo()


def batch_graph(data1, data2, k_numeber):
    coo1 = coo_adj(data1, data2, k_numeber)
    print(coo1.shape)
    coo2 = coo_adj(data2, data1, k_numeber)
    print(coo2.shape)
    return coo_pairs(coo1, coo2)


def purity_score(y_true, y_pred):
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1)


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def consistencyloss_vgae(emb, A):
    mat = torch.sigmoid(cosine_similarity(emb))
    return torch.mul(A, torch.log(mat)).mean()


def consistencyloss_vgae_mosaic(emb, A, B, c, d):
    mat1 = torch.sigmoid(cosine_similarity(emb[c]))
    mat2 = torch.sigmoid(cosine_similarity(emb[d]))

    return torch.mul(A, torch.log(mat1)).mean() + torch.mul(B, torch.log(mat2)).mean()


def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)


def consistencyloss(emb, A):
    emb = emb - torch.mean(emb, dim=0, keepdim=True)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    cov = torch.matmul(emb, emb.t())
    return torch.mean((cov - A.to_dense()) ** 2)


def consistencyloss_new(emb, A):
    cov = torch.matmul(emb, emb.t())
    mat = torch.sigmoid(cov)
    return torch.mean((cov - A.to_dense()) ** 2)


def spatial_construct_graph1(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A = np.zeros((coor.shape[0], coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]] = 1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg  # , nsadj


def spatial_construct_graph(positions, k=15):
    print("start spatial construct graph")
    A = euclidean_distances(positions)
    tmp = 0
    mink = 2
    for t in range(100, 1000, 100):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 100, 1000, 10):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 10, 1000, 5):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            A = A1
            break
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/edge.csv', index, delimiter=',')

    graph_nei = torch.from_numpy(A)
    # print(type(graph_nei),graph_nei)
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg  # , nsadj


def features_construct_graph1(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    from sklearn.metrics import pairwise_distances
    # data,
    # n_components = 50,
    # gene_dist_type = "cosine",
    # ):
    pca = PCA(n_components=50)
    # if isinstance(features, np.ndarray):
    #     data_pca = pca.fit_transform(features)
    # elif isinstance(features, csr_matrix):
    #     data = features.toarray()
    data_pca = pca.fit_transform(features.toarray())
    gene_correlation = 1 - pairwise_distances(data_pca, metric="cosine")
    return gene_correlation

    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/fadj.csv', index, delimiter=',')
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj  # , nfadj


def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/fadj.csv', index, delimiter=',')
    fadj = sp.coo_matrix(A, dtype=np.float32)
    # 可以用来batch间的近邻
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj  # , nfadj


# def load_data(config):
#     print("load data:")
#
#     labels = pd.read_table(config.label_path, sep='\t')
#     labels = labels["layer_guess_reordered"].copy()
#
#     NA_index = np.where(labels.isnull())
#     labels = labels.drop(labels.index[NA_index])
#
#     adata = h5py.File(config.feature_path, 'r')
#     print("path: ",config.feature_path)
#     data = np.array(adata['matrix']["data"])
#     indices = np.array(adata['matrix']["indices"])
#     indptr = np.array(adata['matrix']["indptr"])
#     shape = np.array(adata['matrix']["shape"])
#     res = csr_matrix((data, indices, indptr), shape=[shape[1], shape[0]]).toarray()
#     print("data shape: ", res.shape)
#     res = np.delete(res, NA_index, axis=0)
#     print("The data shape after delete the spots with label nan: ", res.shape)
#
#     # adata = sc.AnnData(res)
#     adata = normalize(res, highly_genes=config.fdim)
#     # print("features: ", adata.X.shape)
#     features = sp.csr_matrix(adata.X, dtype=np.float32)
#     features = torch.FloatTensor(np.array(features.todense()))
#     fadj, nfadj = features_construct_graph(features, k=config.k)
#
#     positions = pd.read_csv(config.positions_path, sep=',')
#     # print("positions shape: ", positions.shape)
#     index_labels = labels.index
#     index_positions = positions.iloc[:, 0]
#     dict = []
#     for i in range(len(index_labels)):
#         index = index_positions[index_positions == index_labels[i]].index[0]
#         dict.append(positions.iloc[index, [4, 5]])
#     positions = np.array(dict, dtype=float)
#     # print("after positions shape: ", positions.shape)
#     # np.savetxt('./result/positions.csv', positions, delimiter=',')
#     sadj, nsadj, graph_nei, graph_neg = spatial_construct_graph(positions, k=config.k1)
#     # print("index_labels: ", index_labels)
#     # np.savetxt('./result/label.csv', ground, delimiter=',')
#
#
#     sadj = torch.LongTensor(sadj.todense())
#     fadj = torch.LongTensor(fadj.todense())
#
#
#     print("done")
#
#
#
#     return features, labels, nsadj, nfadj, graph_nei, graph_neg


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # -*- coding : utf-8-*-
    # coding:unicode_escape

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def get_adj(data, pca=None, k=25, mode="connectivity", metric="cosine"):
    if pca is not None:
        data = dopca(data, dim=pca)
        data = data.reshape(-1, 1)
    A = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    # S = cosine_similarity(data)
    return adj, adj_n  # , S


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


class louvain:
    def __init__(self, level):
        self.level = level
        return

    def updateLabels(self, level):
        # Louvain algorithm labels community at different level (with dendrogram).
        # Here we want the community labels at a given level.
        level = int((len(self.dendrogram) - 1) * level)
        partition = community_louvain.partition_at_level(self.dendrogram, level)
        # Convert dictionary to numpy array
        self.labels = np.array(list(partition.values()))
        return

    def update(self, inputs, adj_mat=None):
        """Return the partition of the nodes at the given level.

        A dendrogram is a tree and each level is a partition of the graph nodes.
        Level 0 is the first partition, which contains the smallest communities,
        and the best is len(dendrogram) - 1.
        Higher the level is, bigger the communities are.
        """
        self.graph = nx.from_numpy_matrix(adj_mat)
        self.dendrogram = community_louvain.generate_dendrogram(self.graph)
        self.updateLabels(self.level)
        self.centroids = computeCentroids(inputs, self.labels)
        return


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([np.mean(data[labels == i], axis=0) for i in range(n_clusters)])


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


class Colors():
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#633194",
        "#8c564b",
        "#F73BAD",
        "#ad494a",
        "#F6E800",
        "#01F7F7",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#8c6d31"]


def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.01):
    '''
                arg1(adata)[AnnData matrix]
                arg2(fixed_clus_count)[int]

                return:
                    resolution[int]
            '''
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):  # , reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            # print(res,' ' , count_unique_leiden)
            if count_unique_leiden == fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag = 0
                break
            if count_unique_leiden > fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag = 1
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):  # , reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            # print(res,' ' , count_unique_louvain)
            if count_unique_louvain == fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 0
                break
            if count_unique_louvain > fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 1
                break
    return cluster_labels, flag


def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)  # 等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC


import torch
import random
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from munkres import Munkres


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


def parameter(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("sum:" + str(k))
    return str(k)


def plot_pca_scatter(name, n_clusters, X_pca, y):
    if name == "usps":
        colors = ['black', 'blue', 'purple', 'yellow', 'pink', 'red', 'lime', 'cyan', 'orange', 'gray']  # usps:10
    elif name == "acm":
        colors = ['yellow', 'pink', 'red']  # acm:3
    elif name == "dblp":
        colors = ['yellow', 'pink', 'red', 'orange']  # dblp:4
    elif name == "cite":
        colors = ['yellow', 'pink', 'red', 'lime', 'cyan', 'orange']  # cite:6
    elif name == "hhar":
        colors = ['green', 'blue', 'red', 'pink', 'yellow', 'purple']  # hhar:6
    elif name == "reut":
        colors = ['green', 'blue', 'red', 'pink']  # reut:4
    else:
        print("Loading Error!")

    for i in range(len(colors)):
        px = X_pca[:, 0][y == i]
        py = X_pca[:, 1][y == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(n_clusters))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    # plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
