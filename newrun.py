# -*- coding: utf8 -*-
import re
import itertools
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from soydata.data.clustering.clustering import make_circular_clusters
from soydata.data.regression import make_stepwise_regression_data
from soydata.data.classification import make_multilayer_rectangulars, make_spiral
from soydata.data import make_rectangular
from soydata.visualize import scatterplot
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs, load_iris, load_wine, load_digits, load_breast_cancer, make_moons, \
    make_circles, make_swiss_roll, make_gaussian_quantiles
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, f1_score, accuracy_score, \
    davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
# from metric_utils import dunn
from validclust import dunn, cop
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sklearn
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# import numpy as np
import pandas as pd
from scipy.io import arff as realarff
import arff    # liac-arff
import time
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from collections import Counter
from cdbw import CDbw
from s_dbw import S_Dbw, SD
# import DBCV
# from fastDBCV import DBCV
# from VIASCKDE import VIASCKDE
# from get_label_map import best_map
import hdbscan
# from silouette_utils import cal_sc, cal_sc2, cal_knnsc, cal_local_sc, inter_k_local_density, inter_k_local_densityV2, \
#     make_clusters, make_clustersV2, check_connect
from silouette_utils import make_clusters, make_clustersV2
from my_metric_utils import GPVIM, GPVI   # CalMyMetric, CalMyMetricContour, GPVI, CalConnectMetric,
# from my_lrd_utils import Cal_LDense
# from my_lrd_utils2 import Cal_Grp_LDense
# from c_index import (calc_c_index, calc_cindex_clusterSim_implementation,
# calc_cindex_nbclust_implementation, pdist_array)
# from othermetric import DSI
# from somecolors import cnames

markers = ["o", "s", "^", "p", "P", "H", "X", "D", "d", "2"]
colors = ["red", "yellow", "blue", "green", "cyan", "pink", "purple", "orange", "sandybrown", "springgreen",
          "mediumslateblue", "olive", "fuchsia", "teal", "grey"]          # Put gray at the end


def load_exist_data(name: str):
    if name == "iris":
        dataset = load_iris()
        x, y = dataset["data"], dataset["target"]
    if name == "wine":       # Last dimension is large
        """Normalization"""
        dataset = load_wine()
        x, y = dataset["data"], dataset["target"]
    if name == "glass":
        """Normalization"""
        dataset = realarff.loadarff("./datasets/real-world/glass.arff")
        # print(dataset)
        data, meta = dataset
        print(data.shape)
        df = pd.DataFrame(data)
        labmaps = {lab: idx for idx, lab in enumerate(np.unique(df.iloc[:, -1]))}
        df.iloc[:, -1] = df.iloc[:, -1].map(labmaps)
        print(df)
        data = df.values
        x, y = data[:, :9].astype(np.float), data[:, -1].astype(np.int)    #

    if name == "insect":
        dataset = arff.load(open("./datasets/artificial/insect.arff"))
        data = np.array(dataset['data'])

        def func(a):
            for i in range(len(a)):
                a[i] = ord(a[i]) - ord('A')
            return a

        x, y = data[:, :2].astype(np.float), np.apply_along_axis(func, 0, data[:, -1]).astype(np.int)
    if name=="haberman":
        dataset = "./datasets/other/haberman.data"
        df = pd.read_csv(dataset, header=None)
        print(df)
        data = df.values
        x, y = data[:, :-1].astype(np.float), data[:, -1].astype(np.int)
    if name == "seed":
        dataset = "./datasets/other/Seed_Data.csv"
        df = pd.read_csv(dataset)
        print(df.head())
        data = df.values
        x, y = data[:, :-1].astype(np.float), data[:, -1].astype(np.int)

    if name=="yeast":
        dataset = "./datasets/other/yeast.data"
        dataurl1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        dataurl2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
        try:
            df = pd.read_csv(dataset, header=None, delim_whitespace=True, index_col=0)  # This dataset requires a specified delimiter
        except:
            df = pd.read_csv(dataurl1, header=None, delim_whitespace=True, index_col=0)
        # finally:
        #     df = pd.read_csv(dataurl2, header=None, delim_whitespace=True, index_col=0)
        # print(df)
        labmaps = {lab: idx for idx, lab in enumerate(np.unique(df.iloc[:, -1]))}
        df.iloc[:, -1] = df.iloc[:, -1].map(labmaps)
        # print(df.iloc[:, 0])
        data = df.values
        x, y = data[:, :-1].astype(np.float), data[:, -1].astype(np.int)
    if name == "chainlink":
        dataset = arff.load(open("./datasets/artificial/chainlink.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :3].astype(np.float), data[:, -1].astype(np.int)
    if name == "3spiral":
        dataset = arff.load(open("./datasets/artificial/3-spiral.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "complex9":
        dataset = arff.load(open("./datasets/artificial/complex9.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "cluto-t8":
        dataset = arff.load(open("./datasets/artificial/cluto-t8-8k.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1]
        y[y == "noise"] = -1
        y = y.astype(np.int)
    if name == "dense-disk-3000":
        dataset = arff.load(open("./datasets/artificial/dense-disk-3000.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "elly-2d10c13s":
        dataset = arff.load(open("./datasets/artificial/elly-2d10c13s.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name=="cure-t2-4k":
        dataset = arff.load(open("./datasets/artificial/cure-t2-4k.arff"))
        data = np.array(dataset['data'])
        def func(a):
            for i in range(len(a)):
                if a[i]=="noise":
                    a[i] = -1
            return a
        x, y = data[:, :2].astype(np.float), np.apply_along_axis(func, 0, data[:, -1]).astype(np.int)
    if name == "rings":
        dataset = arff.load(open("./datasets/artificial/rings.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "zelnik5":
        dataset = arff.load(open("./datasets/artificial/zelnik5.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "zelnik4":
        dataset = arff.load(open("./datasets/artificial/zelnik4.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1]
        y[y == "noise"] = -1
        y = y.astype(np.int)
        # print(y)
    if name == "donutcurves":
        dataset = arff.load(open("./datasets/artificial/donutcurves.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "insect":
        dataset = arff.load(open("./datasets/artificial/insect.arff"))
        data = np.array(dataset['data'])
        def func(a):
            for i in range(len(a)):
                a[i] = ord(a[i]) - ord('A')
            return a
        x, y = data[:, :2].astype(np.float), np.apply_along_axis(func, 0, data[:, -1]).astype(np.int)
    if name == "square4":
        dataset = arff.load(open("./datasets/artificial/square4.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :-1].astype(np.float), data[:, -1].astype(np.int)
    if name == "2d-4c-no4":
        dataset = arff.load(open("./datasets/artificial/2d-4c-no4.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :-1].astype(np.float), data[:, -1].astype(np.int)
    if name == "2sp2glob":
        dataset = arff.load(open("./datasets/artificial/2sp2glob.arff"))
        data = np.array(dataset['data'])
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "combination9":
        dataset = "./datasets/other/combination9.csv"
        df = pd.read_csv(dataset)
        print(df.head())
        data = df.values
        x, y = data[:, :2].astype(np.float), data[:, -1].astype(np.int)
    if name == "clusterable_data":
        data = np.load('./datasets/clusterable_data.npy')
        print(data)
        print(data.shape)
        exit(0)

    # Make sure labels start at 0
    unilab = np.unique(y)
    minilab = min(unilab[unilab != -1])  # The smallest class label, excluding the -1 noise class
    if minilab != 0:
        for i in range(len(y)):
            if y[i] != -1:
                y[i] -= minilab

    return x, y, name


def make_simulation_data(name):
    if name == 'rand':
        x, y = make_blobs(n_samples=50, n_features=3, centers=3, center_box=(-5, 5),
                          cluster_std=0.5)  # A larger center_box interval combined with a smaller standard deviation results in greater data concentration
    elif name == "moon":
        x, y = make_moons(n_samples=500, noise=0.05, random_state=1)
    elif name == "moon-right":
        x, y = make_moons(n_samples=500, noise=0.05, random_state=1)

        """In the case of two classes, relocate one of the classes (e.g., the 'moon' class)"""  
        move = True
        if move:
            for idx, arr in enumerate(x):
                if y[idx] == 1:
                    # arr += 1.5        # inplace, shift along the first and second dimensions toward the top-right
                    arr[0], arr[1] = arr[0]+.5, arr[1]+0    # inplace, move to the right
                    # arr[0], arr[1] = arr[0]+0, arr[1]-.5      # inplace, move to the down
    elif name =="moon2":
        x, y = make_moons(n_samples=50, noise=0.05)
    elif name == "circle":
        x, y = make_circles(n_samples=200, noise=0.02, factor=0.3)  # Smaller factor → greater gap between the two rings
    elif name == "artificial1":
        """
        DBSCAN(eps=0.2, min_samples=5)
        """
        X1, y1 = make_circles(n_samples=500, factor=.6,
                              noise=.03)
        X2, y2 = make_blobs(n_samples=100, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],
                            random_state=9)  # This center setting produces a single class
        y2 = np.array([2] * len(y2))  # Manually set as the third class
        x = np.concatenate((X1, X2))
        y = np.concatenate((y1, y2))
    elif name == "artificial2":
        """
        """
        X1, y1 = make_blobs(n_samples=100, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],
                            random_state=9)  # This center setting produces a single class

        X2, y2 = make_blobs(n_samples=100, n_features=2, centers=[[1.3, 1.3]], cluster_std=[[.2]], random_state=10)
        y2 = np.array([1] * len(y2))  # Manually set as the second class
        x = np.concatenate((X1, X2))
        y = np.concatenate((y1, y2))
    elif name == "artificial3":
        X1, y1 = make_blobs(n_samples=50, n_features=2, centers=[[1.1, 1.1]], cluster_std=[[0.2]], random_state=9)  # This center setting produces a single class
        X2, y2 = make_blobs(n_samples=50, n_features=2, centers=[[2.1, 4.1]], cluster_std=[[0.2]], random_state=10)
        X3, y3 = make_blobs(n_samples=50, n_features=2, centers=[[3.1, 3.1]], cluster_std=[[0.2]], random_state=11)
        X4, y4 = make_gaussian_quantiles(n_samples=80, n_features=2, cov=1.5, mean=[2,3])     # mean is center
        # X4, y4 = make_blobs(n_samples=50, n_features=2, centers=[[2,3]], cluster_std=[[2]], random_state=11)
        y2 = np.array([1] * len(y2))    # Manually set as the second class
        y3 = np.array([2] * len(y3))   # Manually set as the third class
        y4 = np.array([-1] * len(y4))  # Manually set as the noise class
        x = np.concatenate((X1, X2, X3, X4))
        y = np.concatenate((y1, y2, y3, y4))

    elif name == "artificial5":
        X, labels = make_circular_clusters(n_clusters=10, r_min=0.05, r_max=0.15,
                                           equal_density=True, noise=0.05, seed=0)
        x, y = X, labels
    elif name == "artificial5-move":
        df = pd.read_csv("./datasets/other/artificial5.csv")
        print(df)
        x, y = df.values[:, :-1].astype(np.float), df.values[:, -1].astype(np.int)
        for idx, arr in enumerate(x):
            if y[idx] == 4:
                arr[0], arr[1] = arr[0]-0.05, arr[1]+0
            if y[idx] == 5:
                arr[0], arr[1] = arr[0]-0.06, arr[1]+0.05
            if y[idx] == 9:
                arr[0], arr[1] = arr[0]+0.16, arr[1]+0
            if y[idx] == 8:
                arr[0], arr[1] = arr[0]-0.25, arr[1]+0.02
            if y[idx] == 6:
                arr[0], arr[1] = arr[0]-0.03, arr[1]+0.27
            if y[idx] == 7:
                arr[0], arr[1] = arr[0]+0, arr[1]+0.02

        # exit(0)

    return x, y, name


def guiyi(X, name: str):
    if name == "MinMaxScaler":
        scaler = MinMaxScaler()
    if name == "StandardScaler":
        scaler = StandardScaler()
    new_X = scaler.fit_transform(X)
    return new_X





















class RUN:
    def __init__(self):
        self.ans_dict = dict()
    
    def fit_clusters(self, data, data_name, algorithm, args, kwds):

        pre_labels = algorithm(*args, **kwds).fit_predict(data)   # if dont use ground true label for external index
        # pre_labels = y                       # test really class     if provide ground true labels for external index

        pred_y = pre_labels
        # print(f"Predicted class: {pred_y}")
        counter = Counter(pred_y)
        print(f"The number of clusters predicted:{counter}")


        return pred_y, self.ans_dict

    def my_metric(self, x, pred_y,  y=None):

        """# Process once"""
        oricluster, orilabel, _, _ = make_clusters(x, y)
        # newcluster2draw, newlabel2draw, newcluster2cal, newlabel2cal = make_clusters(x, pred_y)
        newcluster2draw, newlabel2draw, newcluster2cal, newlabel2cal, resp_ori_idxes, resp_ori_idxes2cal = make_clustersV2(x, pred_y)
        newcluster2cal, newlabel2cal = newcluster2draw, newlabel2draw
        resp_ori_idxes2cal = resp_ori_idxes
        # newx = np.concatenate(newcluster2cal)

        """GPVIM"""
        try:
            calgpvim = GPVIM(newcluster2cal, newlabel2cal, resp_ori_idxes2cal, len(x))
            wholegpvim, gpvims_pointdis, gpvim_time = calgpvim.cal_gpvim()
            gpvims, gpvimpointdis = gpvims_pointdis[0], gpvims_pointdis[1]
            # print(f"gpvim of each cluster:{[np.mean(item) for item in gpvims]}")

            print(f"Total GPVIM:{wholegpvim}")
            self.ans_dict["gpvim"] = {"time": gpvim_time, "score":wholegpvim}
        except:
            self.ans_dict["gpvim"] = {"time": np.NAN, "score": np.NAN}
        print("*" * 50)

        """GPVI"""
        try:
            calgpvi = GPVI(newcluster2cal, newlabel2cal, resp_ori_idxes2cal, len(x))
            wholegpvi, gpvis, gpvi_time = calgpvi.cal_gpvi()
            print(f"Total GPVI: {wholegpvi}")
            self.ans_dict["gpvi"] = {"time":gpvi_time, "score":wholegpvi}
        except:
            self.ans_dict["gpvi"] = {"time":np.NAN, "score":np.NAN}
        print("*" * 50)




if __name__=="__main__":
    """load data"""
    x, y, dataname = make_simulation_data('moon')
    # x, y, dataname = load_exist_data("zelnik4")
    print(f"Data: {dataname}, samples: {x.shape[0]}, features: {x.shape[1]}, class: {np.unique(y)}")

    """Extreme data 1"""
    # x, y, dataname = np.array([
    #            [0, 9],
    #            [0, 9.5],
    #            [7, 0],
    #            [7.5, 0],
    #            [3, 3],
    #            [3, 0],
    #     [0, 9],
    #     [0, 9.5],
    #     [7, 0],
    #     [7.5, 0],
    #     [3, 3],
    #     [3, 0]]), np.array([0,0,0,0,0,0,1,1,1,1,1,1]), 'check'

    """Extreme data 2"""
    # x, y, dataname = np.array([[0, 9],
    #                            [0, 9.5],
    #                            [7, 0],
    #                            [7.5, 0],
    #                            [3, 3],
    #                            [3, 0],
    #                     [0,9],
    #                     [0,9.5],
    #                     [7,0],
    #                     [7.5,0],
    #                     [3,3],
    #                     [3,0]
    #
    #                         ]), np.array([0,0,0,0,0,0,1,1,1,1,1,1]), "check"

    """Extreme data 3"""
    # x, y, dataname = np.array([[0, 1],
    #                            [0.2, 1.5],
    #                            [0.25, 1.25],
    #                            [0.1, 1.1],
    #                            [0.2, 1.32],
    #                            [0.3, 1.4],
    #                        [0, 1],
    #                        [0.2, 1.5],
    #                        [0.25, 1.25],
    #                        [0.1, 1.1],
    #                        [0.2, 1.32],
    #                        [0.3, 1.4],
    #
    #                            ]), np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]), "check"
    #
    # print(x.shape)

    """Normalization"""
    # x = guiyi(x, name="StandardScaler")  # MinMaxScaler  StandardScaler

    # print(f"Ground True label：{y}")

    runner = RUN()
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, sklearn.cluster.KMeans, (), {'n_clusters':2})
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, KMedoids, (), {'n_clusters': 2, 'method':'pam'})
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, hdbscan.HDBSCAN, (), {'min_cluster_size':15, 'min_samples':6})
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps":0.5, "min_samples":10})        # circle
    pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps":0.37254, "min_samples":28})      # moon
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps": 0.2, "min_samples": 28})      # moon right,zelnik4
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps": 0.5, "min_samples": 10})    # 2sp2glob
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps": 0.3, "min_samples": 10})  # 2sp2glob
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps": 3, "min_samples": 10})  # 3spiral
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps": 0.03, "min_samples": 6})  # artificial1
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, DBSCAN, (), {"eps": 0.3, "min_samples": 6})  # glass
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, hdbscan.HDBSCAN, (), {"min_cluster_size":10})   # digits, yeast, moon right
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, hdbscan.HDBSCAN, (),
    #                                        {"min_cluster_size": 10, 'min_samples':10})  # iris
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, GaussianMixture, (), {"n_components":6, "covariance_type":'full'})
    # pred_y, ans_dict = runner.fit_clusters(x, dataname, SpectralClustering, (), {"n_clusters": 2})
    # pred_y = y
    runner.my_metric(x=x, pred_y=pred_y, y=y)

    # plt.grid()
    # plt.show()

