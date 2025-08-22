# -*- coding: utf8 -*-
import math

from scipy.spatial.distance import euclidean
# from lrd_utils import get_lrdensity, get_inter_lrdensity
from sklearn.datasets import make_blobs, load_iris, make_moons, make_circles
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, f1_score, accuracy_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from validclust import dunn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
# import torch
import functools


def time_count(func):
    def run(*args, **kwargs):
        start = time.time()
        score, allscore = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time of {func.__name__}: {end - start:.3f}s")
        return score, allscore, end-start
    return run



def guiyi(X, name:str):
    if name=="MinMaxScaler":
        scaler = MinMaxScaler()
    new_X = scaler.fit_transform(X)
    return new_X



def make_clusters(data, label):
    import copy
    clusters, clusters2cal = [], None
    index = None
    unilab = np.unique(label)
    for i,lab in enumerate(unilab):
        # if lab!=-1:    # Non-noise points
        #     clusters2cal.append(data[np.where(label == lab)[0]])
        if lab==-1:
            index = i
        clusters.append(data[np.where(label == lab)[0]])
    newlabel, newlabel2cal = [], []
    for idx, clus in enumerate(clusters):
        # if unilab[idx]!=-1:
        #     newlabel2cal.append([unilab[idx]] * len(clus))
        newlabel.append([unilab[idx]] * len(clus))  # extend

    if index is not None:
        clusters2cal = copy.deepcopy(clusters)
        clusters2cal.pop(index)
        newlabel2cal = copy.deepcopy(newlabel)
        newlabel2cal.pop(index)
    else:
        clusters2cal = clusters
        newlabel2cal = newlabel
    return clusters, newlabel, clusters2cal, newlabel2cal


def make_clustersV2(data, label):
    import copy
    clusters, clusters2cal = [], None
    respect_ori_indexes = []
    index = None
    unilab = np.unique(label)
    for i,lab in enumerate(unilab):
        # if lab!=-1:    # Non-noise points
        #     clusters2cal.append(data[np.where(label == lab)[0]])
        if lab==-1:      # Check for noise class
            index = i
        clusters.append(data[np.where(label == lab)[0]])
        respect_ori_indexes.append(np.where(label == lab)[0])
    newlabel, newlabel2cal = [], []
    for idx, clus in enumerate(clusters):
        # if unilab[idx]!=-1:
        #     newlabel2cal.append([unilab[idx]] * len(clus))
        newlabel.append([unilab[idx]] * len(clus))  # extend

    if index is not None:   # noise class
        clusters2cal = copy.deepcopy(clusters)
        clusters2cal.pop(index)
        newlabel2cal = copy.deepcopy(newlabel)
        newlabel2cal.pop(index)
        respect_ori_indexes2cal = copy.deepcopy(respect_ori_indexes)
        respect_ori_indexes2cal.pop(index)
    else:
        clusters2cal = clusters
        newlabel2cal = newlabel
        respect_ori_indexes2cal = respect_ori_indexes
    return clusters, newlabel, clusters2cal, newlabel2cal, respect_ori_indexes, respect_ori_indexes2cal




def topk_(matrix, K, axis=1):
    """
    Like PyTorch topk: select the top-k largest values
    :param matrix:
    :param K:
    :param axis:
    :return:
    """
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        # topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_index = np.argpartition(-matrix, K-1, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        # topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_index = np.argpartition(-matrix, K-1, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort


if __name__ == '__main__':
    x = np.array([[2,0,1],
                  [4,2,3]])
    val, idx = topk_(x, K=3, axis=1)
    print(f"val:\n{val}")
    print(f"idx:\n{idx}")





