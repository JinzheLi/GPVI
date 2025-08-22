import functools
from typing import Tuple, Dict, List
from sklearn.datasets import make_moons, make_blobs, make_circles, make_gaussian_quantiles
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, euclidean
from silouette_utils import make_clusters, make_clustersV2, topk_, time_count
# import DBCV
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import itertools
import statistics
import math
from collections import defaultdict as ddict
from relativeNeighborhoodGraph import returnRNG





class GPVIM:
    def __init__(self, clusters, label, respect_ori_indexes, length):
        """

        :param clusters: clustered data
        :param label: Cluster labels
        :param respect_ori_indexes:
        :param length: Dataset size
        """
        self.clusters = clusters
        self.label = label
        self.respect_ori_indexes = respect_ori_indexes
        self.length = length

        self.wholestd = np.linalg.norm(np.std(np.concatenate(clusters, axis=0)))

    def weighted_std(self, xs, ws):
        """
        Compute the weighted standard deviation
        :param xs:
        :param ws:
        :return:
        """
        xbarw = np.sum(xs*ws) / np.sum(ws)
        fenzi = np.sum(ws*np.square(xs-np.array([xbarw]*len(xs))))
        fenmu = ((sum(ws!=0)-1) * np.sum(ws)) / sum(ws!=0)
        ans = np.sqrt(fenzi/fenmu)
        return ans

    def left_composes2chosencomp(self, curclusterlen, subgs, suba, suba_pos, subb=None, subb_pos=None):
        """
        Mean minimum distance from other components to the chosen component pair
        :param curclusterlen: Number of samples in the current cluster
        :param subgs: Set of subgraphs
        :param suba: Subgraph 1
        :param suba_pos: Samples of Subgraph 1
        :param subb: Subgraph 2
        :param subb_pos: Samples of Subgraph 2
        :return:
        """
        lf_comp_betwn_dsts = []
        if suba and subb:
            sub_concat_cluster = np.concatenate([suba_pos, subb_pos], axis=0)
            for subgph, subgphspls in subgs:
                if subgph not in [suba, subb]:
                    tmpmin = np.min(cdist(subgphspls, sub_concat_cluster))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # Square rooting
        elif not subb:
            for subgph, subgphspls in subgs:
                if subgph not in [suba]:
                    tmpmin = np.min(cdist(subgphspls, suba_pos))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # Square rooting
        else: raise ValueError
        lf_comp_betwn_dsts = np.array(lf_comp_betwn_dsts)
        # print(f"lf_comp_betwn_dsts: {lf_comp_betwn_dsts}")
        # lf_comp_betwn_avg = np.average(lf_comp_betwn_dsts[:, 0], weights=lf_comp_betwn_dsts[:, 1])
        lf_comp_betwn_avg = np.einsum('i,i->', lf_comp_betwn_dsts[:, 0], lf_comp_betwn_dsts[:, 1]) if len(lf_comp_betwn_dsts)!=0 else 0
        # print(f"lf_comp_betwn_avg: {lf_comp_betwn_avg}")
        return lf_comp_betwn_avg

    def gen_graph(self, data, ori_idxs, label=None, figsize=None):
        cdistance = cdist(data, data)
        edges = [[(ori_idxs[i], ori_idxs[j], round(w, 3)) for j, w in enumerate(line)] for i, line in enumerate(cdistance)]
        edges = list(itertools.chain.from_iterable(edges))
        # print(f"origin Graph edges: {edges}")
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        # pos = {idx: coor for idx, coor in enumerate(data)}
        pos = {ori_idxs[idx]: coor for idx,coor in enumerate(data)}

        # plot figure
        # plt.figure(figsize=figsize)
        # nx.draw(G, pos=pos, with_labels=True, node_color='c', alpha=0.8)

        return G, pos, cdistance

    def gen_concat_graph(self, concatdata: np.ndarray, concat_ori_idx: np.ndarray):
        """

        :param concatdata: 2-d array
        :param concat_ori_idx: 1-d array
        :return:
        """
        cdistance = cdist(concatdata, concatdata)
        edges = [[(concat_ori_idx[i], concat_ori_idx[j], round(w, 3)) for j, w in enumerate(line)] for i, line in
                 enumerate(cdistance)]
        edges = list(itertools.chain.from_iterable(edges))
        # print(f"origin Graph edges: {edges}")
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        # pos = {idx: coor for idx, coor in enumerate(concatdata)}
        pos = {concat_ori_idx[idx]:coor for idx, coor in enumerate(concatdata)}

        return G, pos

    def gen_mst(self, G, pos=None, data=None, label=None, figsize=None):
        """

        :param G:
        :param pos:
        :param data:
        :param label:
        :param figsize:
        :return: Minimum Spanning Tree: nx.classes.graph, edge weights of the MST:dict
        """

        G_MST = nx.minimum_spanning_tree(G, weight='weight')
        w = nx.get_edge_attributes(G_MST, 'weight')
        # print(f"MST weight: {w}")

        """
        # Plot the minimum spanning tree
        # plt.rcParams['figure.figsize'] = (8, 8)  # Set the figure size
        plt.figure(figsize=figsize)
        nx.draw(G_MST, pos=pos, with_labels=True, node_color='c')
        nx.draw_networkx_edges(G, pos=pos, edgelist=G_MST.edges, edge_color='orange')
        nx.draw_networkx_edge_labels(G_MST, pos=pos, edge_labels=w, font_color='m')
        # plt.show()
        """

        return G_MST, w


    def gen_RNG(self, g_pos, distance_matrix, figsize=None):
        """

        :param g_pos:
        :param distance_matrix:
        :param figsize:
        :return: Return a graph in the same format as NetworkX.
        """
        RNGdf = returnRNG.returnRNG(distance_matrix)
        print(RNGdf)
        RNG = nx.from_pandas_adjacency(RNGdf)

        # plot figure
        # plt.figure(figsize=figsize)
        # nx.draw(RNG, pos=g_pos, with_labels=True, node_color='c', alpha=0.8)

        return RNG


    def cal_cluster_intraV2(self, cluster, cur_cluster_ori_idxs) :
        """
        Updates over V7/V8:
        1) Component pairing:
        - When selecting a component pair, use the two components incident to the cutting edge.

        2) Intra-class distance computation:
        (a) Within-component distance:
            - Construct an RNG or MST for each component.
            - Compute the mean edge length as the within-component distance.
            - Aggregate across components via a weighted average with weight:
                w_intra(sub) = sub.number_of_nodes() / len(cluster).

        (b) Between-component distance:
            - Use the edges directly connecting two components (inter-component edges).
            - Compute the mean length over these connecting edges.
            - Aggregate via a weighted average with weight:
                w_inter(subA, subB) = subA.number_of_nodes() * subB.number_of_nodes() / (len(cluster) ** 2).

        3) Final distance:
        - Take the (unweighted) mean of the within-component and between-component results.
        :param cluster:ndarray[n_sample, m_feature] cluster
        :return:
        """
        if len(cluster)==1:   # Case when the cluster has only a single sample
        # print(f"Cluster with only one sample, shape: {cluster.shape}")
            return [0]

        cluster_G, cluster_pos, cluster_dst = self.gen_graph(cluster, cur_cluster_ori_idxs)
        cluster_MST, cluster_MST_w = self.gen_mst(G=cluster_G, pos=cluster_pos)
        # print(f"cluster_MST_w: {cluster_MST_w}")
        # print(f"The MST average of this cluster is: {np.mean(list(cluster_MST_w.values()))}")
        # cluster_MST_nodes = cluster_MST.nodes()

        """Contour points"""
        contour_node_number = np.array([x for x in cluster_MST.nodes() if cluster_MST.degree(x) == 1])
        # print(f"Count of contour points: {len(contour_node_number)}\nList of contour points: {contour_node_number}")
        # Method 1
        # contour_sample = cluster[np.where(cur_cluster_ori_idxs == contour_node_number[:, None])[-1]]
        # Method 2
        # contour_sample = np.concatenate([np.expand_dims(node_pos, axis=0) for node_num,node_pos in cluster_pos.items() if node_num in contour_node_number], axis=0)
        # Method 3
        contour_sample = []
        for node_num in contour_node_number:
            if node_num in cluster_pos:
                contour_sample.append(np.expand_dims(cluster_pos[node_num], axis=0))
        contour_sample = np.concatenate(contour_sample, axis=0)
        # print(f"length of contour_node_number: {len(contour_node_number)}, length of contour_sample: {len(contour_sample)}")

        """Locate the maximum-weight edge in the MST, append its incident vertices to the contour, and perform a cut by removing this edge."""
        sorted_w_of_MST = sorted([item for item in cluster_MST_w.items()], key=lambda x: x[1], reverse=True)
        # print(f"Edges sorted by length (descending): {sorted_w_of_MST}")
        max_edge = sorted_w_of_MST[0][0]
        # print(f"The maximum edge is {max_edge}, and its weight (length) is {sorted_w_of_MST[0][1]}")

        # # Edges > mean + 3*std
        biggeredges = {max_edge:sorted_w_of_MST[0][1]}
        edge_avg, edge_std = np.mean(list(cluster_MST_w.values())), np.std(list(cluster_MST_w.values()))
        for edge, w in cluster_MST_w.items():
            if w > edge_avg + 3 * edge_std:   # 3*sigma
                biggeredges[edge] = w

        # for time in range(3, 0, -1):          # Test the effect using 1-sigma, 2-sigma, and 3-sigma thresholds
        #     for edge, w in cluster_MST_w.items():
        #         if w > edge_avg + time * edge_std:
        #             biggeredges[edge] = w
        #     if biggeredges:
        #         break
        # print(f"biggeredges: {biggeredges}")


        # Prune MST edges > mean + 3*std
        for tocutedge, w in biggeredges.items():
            # print(tocutedge)
            cluster_MST.remove_edge(*tocutedge)

        # Get subgraphs after cutting
        subgs_node2avgedge = []
        subgs = []
        compintras = []  # Record the within-component distance along with the relative proportion of the component
        sparse_sum = 0
        for connect_item in nx.connected_components(cluster_MST):
            subg = cluster_MST.subgraph(connect_item)
            # print(f"subg: {subg.adj}")
            subg_nodes = np.array([x for x in subg.nodes()])  # or list(subg)
            subg_pos = cluster[np.where(cur_cluster_ori_idxs == subg_nodes[:, None])[-1]]   # samples
            # subg_pos2draw = {subg_nodes[idx]:coor for idx, coor in enumerate(subg_pos)}     # plt
            # self.graph_plot(subg, subg_pos2draw)
            subgs.append([subg, subg_pos])

            if subg.number_of_nodes() == 1:
                # print(f"The number of nodes in the subgraph is {1}")
                subgs_node2avgedge.append((1, 0))
                compintras.append([0, subg.number_of_nodes()/cluster_G.number_of_nodes()])
                continue
            subg_w = nx.get_edge_attributes(subg, 'weight')
            # print(f"subg_w: {subg_w}")
            subg_w_mean = np.mean(list(subg_w.values()))
            # print(f"subg_w_mean: {subg_w_mean}")
            subgs_node2avgedge.append((subg.number_of_nodes(), subg_w_mean))
            # sparse_sum += subg_w_mean
            compintras.append([subg_w_mean, subg.number_of_nodes()/cluster_G.number_of_nodes()])
            # print(f"The number of nodes in the subgraph is {subg.number_of_nodes()}")
        # print(f"compintras: {compintras}")

        sparse_tmps = []
        compinters = []     # Save inter-component distance and pair proportion
        curclusterdict = dict()     # Record each point's intra-cluster k-NN distance in the current cluster
        records, coefs = [], []
        tmp_ds2cmps = []
        # -*- Select the component pair at the endpoints of the cutting edge -*-
        for idx, (tocutedge, cutw) in enumerate(biggeredges.items()):
            subA, subB = None, None
            for item in subgs:
                if tocutedge[0] in item[0].nodes():
                    subA, subA_pos = item[0], item[1]
                elif tocutedge[1] in item[0].nodes():
                    subB, subB_pos = item[0], item[1]

                if subA and subB: break   # Indicates that a component pair has been identified
            # print(f"Pair {idx + 1} of components")
            # print(f"The cutting edge is {tocutedge}, with length {cutw}")
            # print(f"subA_pos shape: {subA_pos.shape}, subB_pos shape: {subB_pos.shape}")


            # Visualize the component graph (only applicable for 2-dimensional datasets)
            subAnodes, subBnodes = list(subA), list(subB)
            subApos2draw = {subAnodes[idx]: coor for idx, coor in enumerate(subA_pos)}
            subBpos2draw = {subBnodes[idx]: coor for idx, coor in enumerate(subB_pos)}
            # self.graph_plot(subA, subApos2draw)
            # self.graph_plot(subB, subBpos2draw)

            # Method 0
            # -*- Computation within components -*-
            compinters.append([cutw, len(subAnodes)*len(subBnodes)/cluster_G.number_of_nodes()**2])
        # print(f"compinters: {compinters}")

        compintras = np.array(compintras)
        comp_intra = np.average(compintras[:, 0], weights=compintras[:, 1])
        # print(f"Cluster intra-component distance: {comp_intra}")
        compinters = np.array(compinters)
        comp_inter = np.average(compinters[:, 0], weights=compinters[:, 1])
        # print(f"Cluster intra-component distance: {comp_inter}")
        # sparse = np.sum([comp_intra, comp_inter])
        sparse = np.mean([comp_intra, comp_inter])
        sparse_list = [sparse]*len(cluster)

        return sparse_list


    @time_count
    def cal_gpvim(self):
        """
        cal
        :return:
        """
        inter_mat = np.zeros((len(self.clusters), len(self.clusters)))  # Maintain an inter-cluster dispersion matrix
        inter_mat.fill(np.inf)
        gpvis = []
        # intra_sparses, comps = [], []
        lengths = []
        all_point_dis = []
        all_s = []
        for idx, cur_cluster in enumerate(self.clusters):
            intra_sparses = self.cal_cluster_intraV2(cur_cluster, self.respect_ori_indexes[idx])  # Retrieve the intra-cluster k-NN distance for each point in the current cluster
            # intra_sparses.append(intra_sparse)
            assert len(cur_cluster) == len(intra_sparses)
            # print(f"MLI intra_sparses of all points in the current cluster: {intra_sparses}")
            lengths.append(len(cur_cluster))
            pointdis = []
            tmp_all_s = []
            for i, instance in enumerate(cur_cluster):
                bi = float("inf")
                for j, clusterB in enumerate(self.clusters):
                    if idx == j: continue
                    intercdst = cdist(np.expand_dims(instance, axis=0), clusterB)
                    """Compute the global inter-cluster distance for the current point"""
                    # tmp_inter_dis = np.mean(intercdst)
                    """Compute the k-NN inter-cluster distance for the current point"""
                    interk = int(np.sqrt(len(clusterB)))
                    vals2B, idxs2B = topk_(-intercdst, K=interk, axis=1)
                    vals2B = -vals2B  # Negation
                    tmp_inter_dis = np.sum(vals2B, axis=1).item() / (interk - 1) if interk>1 else np.mean(vals2B)
                    if tmp_inter_dis < bi:
                        bi = tmp_inter_dis
                # print(f"MTL intra-cluster distance for the current point: {intra_sparses[i]}")
                # print(f"MTL inter-cluster distance for the current point: {bi}")
                pointdis.append((intra_sparses[i], bi))
                si = (bi - intra_sparses[i]) / max(intra_sparses[i], bi)
                if si==np.nan:
                    raise ValueError
                tmp_all_s.append(si)
            all_s.append(tmp_all_s)
            all_point_dis.append(pointdis)

        new_all_s = []
        cluster_s = []
        for item in all_s:
            cluster_s.append(np.mean(item))
            new_all_s.extend(item)
        # print(f"type of new_all_s: {type(new_all_s)}")
        sc = np.mean(new_all_s)  # mean
        # print(f"new_all_s: {new_all_s}")
        # print(f"sc: {sc}")

        # Check if the counts are equal
        for i, clu in enumerate(self.clusters):
            # print(f"len of clu: {len(clu)}")
            # print(f"len of sc: {len(all_s[i])}")
            assert len(clu) == len(all_s[i])



        return sc, (all_s, all_point_dis)

    def graph_plot(self, g, g_pos, figsize=None):
        # Visualize the MST with edge weights
        plt.figure(figsize=None)
        w = nx.get_edge_attributes(g, 'weight')
        # print(f"w: {w}")
        nx.draw(g, pos=g_pos, with_labels=True, node_color='c')
        nx.draw_networkx_edges(g, pos=g_pos, edgelist=g.edges, edge_color='orange')
        nx.draw_networkx_edge_labels(g, pos=g_pos, edge_labels=w, font_color='m')
        plt.show()

        return w


class GPVI:
    def __init__(self, clusters, label, respect_ori_indexes, length):
        """
        :param clusters: Clustered data (e.g., a list/array of per-cluster samples or indices).
        :param label: Cluster labels (cluster assignments for the samples).
        :param respect_ori_indexes: Whether to respect/retain the original sample indices (bool).
        :param length: Dataset size (the total number of samples).
        """
        self.clusters = clusters
        self.label = label
        self.respect_ori_indexes = respect_ori_indexes
        self.length = length

        self.wholestd = np.linalg.norm(np.std(np.concatenate(clusters, axis=0)))

    def weighted_std(self, xs, ws):
        """
        Calculate the weighted standard deviation
        :param xs:
        :param ws:
        :return:
        """
        xbarw = np.sum(xs * ws) / np.sum(ws)
        fenzi = np.sum(ws * np.square(xs - np.array([xbarw] * len(xs))))
        fenmu = ((sum(ws != 0) - 1) * np.sum(ws)) / sum(ws != 0)
        ans = np.sqrt(fenzi / fenmu)
        return ans

    def left_composes2chosencomp(self, curclusterlen, subgs, suba, suba_pos, subb=None, subb_pos=None):
        """
        Compute the mean of the minimum distances from the other components to the selected pair of components. 
        :param curclusterlen: Number of samples in the current cluster
        :param subgs: Set of subgraphs
        :param suba: Subgraph 1
        :param suba_pos: Samples of Subgraph 1
        :param subb: Subgraph 2
        :param subb_pos: Samples of Subgraph 2
        :return:
        """
        lf_comp_betwn_dsts = []
        if suba and subb:
            sub_concat_cluster = np.concatenate([suba_pos, subb_pos], axis=0)
            for subgph, subgphspls in subgs:
                if subgph not in [suba, subb]:
                    tmpmin = np.min(cdist(subgphspls, sub_concat_cluster))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # square
        elif not subb:
            for subgph, subgphspls in subgs:
                if subgph not in [suba]:
                    tmpmin = np.min(cdist(subgphspls, suba_pos))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # square
        else:
            raise ValueError
        lf_comp_betwn_dsts = np.array(lf_comp_betwn_dsts)
        # print(f"lf_comp_betwn_dsts: {lf_comp_betwn_dsts}")
        # lf_comp_betwn_avg = np.average(lf_comp_betwn_dsts[:, 0], weights=lf_comp_betwn_dsts[:, 1])
        lf_comp_betwn_avg = np.einsum('i,i->', lf_comp_betwn_dsts[:, 0], lf_comp_betwn_dsts[:, 1]) if len(
            lf_comp_betwn_dsts) != 0 else 0
        # print(f"lf_comp_betwn_avg: {lf_comp_betwn_avg}")
        return lf_comp_betwn_avg

    def gen_graph(self, data, ori_idxs, label=None, figsize=None):
        cdistance = cdist(data, data)
        edges = [[(ori_idxs[i], ori_idxs[j], round(w, 3)) for j, w in enumerate(line)] for i, line in
                 enumerate(cdistance)]
        edges = list(itertools.chain.from_iterable(edges))
        # print(f"origin Graph edges: {edges}")
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        # pos = {idx: coor for idx, coor in enumerate(data)}
        pos = {ori_idxs[idx]: coor for idx, coor in enumerate(data)}

        # plot
        # plt.figure(figsize=figsize)
        # nx.draw(G, pos=pos, with_labels=True, node_color='c', alpha=0.8)

        return G, pos, cdistance

    def gen_concat_graph(self, concatdata: np.ndarray, concat_ori_idx: np.ndarray):
        """

        :param concatdata: 2-d array
        :param concat_ori_idx: 1-d array
        :return:
        """
        cdistance = cdist(concatdata, concatdata)
        edges = [[(concat_ori_idx[i], concat_ori_idx[j], round(w, 3)) for j, w in enumerate(line)] for i, line in
                 enumerate(cdistance)]
        edges = list(itertools.chain.from_iterable(edges))
        # print(f"origin Graph edges: {edges}")
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        # pos = {idx: coor for idx, coor in enumerate(concatdata)}
        pos = {concat_ori_idx[idx]: coor for idx, coor in enumerate(concatdata)}

        return G, pos

    def gen_mst(self, G, pos=None, data=None, label=None, figsize=None):
        """

        :param G:
        :param pos:
        :param data:
        :param label:
        :param figsize:
        :return: Minimum Spanning Tree: nx.classes.graph, MST edge weights: dict
        """

        G_MST = nx.minimum_spanning_tree(G, weight='weight')
        w = nx.get_edge_attributes(G_MST, 'weight')
        # print(f"MST weight: {w}")

        """
        # MST
        # plt.rcParams['figure.figsize'] = (8, 8)  
        plt.figure(figsize=figsize)
        nx.draw(G_MST, pos=pos, with_labels=True, node_color='c')
        nx.draw_networkx_edges(G, pos=pos, edgelist=G_MST.edges, edge_color='orange')
        nx.draw_networkx_edge_labels(G_MST, pos=pos, edge_labels=w, font_color='m')
        # plt.show()
        """

        return G_MST, w

    def gen_RNG(self, g_pos, distance_matrix, figsize=None):
        """

        :param g_pos:
        :param distance_matrix:
        :param figsize:
        :return: # Return a graph in the same format as NetworkX
        """
        RNGdf = returnRNG.returnRNG(distance_matrix)
        print(RNGdf)
        RNG = nx.from_pandas_adjacency(RNGdf)

        # plot
        # plt.figure(figsize=figsize)
        # nx.draw(RNG, pos=g_pos, with_labels=True, node_color='c', alpha=0.8)

        return RNG


    def cal_cluster_intraV2(self, cluster, cur_cluster_ori_idxs):
        """
        different compared with V7/V8:
        - For component pairing, use the two components at the endpoints of the cutting edge.
        - For intra-cluster computation:
        1. Within each component: build an RNG or MST, compute the mean edge length 
            as the intra-component distance, then take a weighted average across components 
            with weight = sub.number_of_nodes() / len(cluster).
        2. Between components: directly use the connecting edges, compute their mean length, 
            then take a weighted average with weight = subA.number_of_nodes() * subB.number_of_nodes() / (len(cluster) ** 2).
        - Finally, take the average of (1) and (2).

        :param cluster: ndarray[n_samples, n_features], a given cluster
        :return:
        """
        if len(cluster)==1:   # Case when the cluster has only one sample
            # print(f"Cluster has only one sample, shape: {cluster.shape}")
            return 0, cluster

        cluster_G, cluster_pos, cluster_dst = self.gen_graph(cluster, cur_cluster_ori_idxs)
        cluster_MST, cluster_MST_w = self.gen_mst(G=cluster_G, pos=cluster_pos)
        # print(f"cluster_MST_w: {cluster_MST_w}")
        # print(f"Cluster MST avg 是: {np.mean(list(cluster_MST_w.values()))}")
        # cluster_MST_nodes = cluster_MST.nodes()

        """Retrieve contour points as well"""
        contour_node_number = np.array([x for x in cluster_MST.nodes() if cluster_MST.degree(x) == 1])
        # print(f"Number of contour points: {len(contour_node_number)}\n Contour points: {contour_node_number}")
        # Method 1
        contour_sample = cluster[np.where(cur_cluster_ori_idxs == contour_node_number[:, None])[-1]]
        # Method 2
        # contour_sample = np.concatenate([np.expand_dims(node_pos, axis=0) for node_num,node_pos in cluster_pos.items() if node_num in contour_node_number], axis=0)
        # Method 3
        # contour_sample = []
        # for node_num in contour_node_number:
        #     if node_num in cluster_pos:
        #         contour_sample.append(np.expand_dims(cluster_pos[node_num], axis=0))
        # contour_sample = np.concatenate(contour_sample, axis=0)
        # print(f"length of contour_node_number: {len(contour_node_number)}, "
        #       f"length of contour_sample: {len(contour_sample)}")

        """Locate the maximum-weight edge in the MST, insert its two endpoints into the contour, and cut the MST by removing this edge"""
        sorted_w_of_MST = sorted([item for item in cluster_MST_w.items()], key=lambda x: x[1], reverse=True)
        # print(f"Edges sorted in descending order of length: {sorted_w_of_MST}")
        max_edge = sorted_w_of_MST[0][0]
        # print(f"The corresponding maximum edge is: {max_edge}, length: {sorted_w_of_MST[0][1]}")

        # Select the edges whose weights are greater than (avg + 3 * standard deviation)
        biggeredges = {max_edge: sorted_w_of_MST[0][1]}
        edge_avg, edge_std = np.mean(list(cluster_MST_w.values())), np.std(list(cluster_MST_w.values()))
        for edge, w in cluster_MST_w.items():
            if w > edge_avg + 3 * edge_std:  # only 3*sigma
                biggeredges[edge] = w

        # for time in range(3, 0, -1):          # try to 1,2,3 sigma
        #     for edge, w in cluster_MST_w.items():
        #         if w > edge_avg + time * edge_std:
        #             biggeredges[edge] = w
        #     if biggeredges:
        #         break
        # print(f"biggeredges: {biggeredges}")

        # Prune MST edges > mean + 3*std
        for tocutedge, w in biggeredges.items():
            # print(tocutedge)
            cluster_MST.remove_edge(*tocutedge)

        # Get the subgraphs after cutting
        subgs_node2avgedge = []
        subgs = []
        compintras = []  # Save intra-component distance and component ratio
        sparse_sum = 0
        for connect_item in nx.connected_components(cluster_MST):
            subg = cluster_MST.subgraph(connect_item)
            # print(f"subg: {subg.adj}")
            subg_nodes = np.array([x for x in subg.nodes()])  # or list(subg)
            subg_pos = cluster[np.where(cur_cluster_ori_idxs == subg_nodes[:, None])[-1]]  # it is samples
            # subg_pos2draw = {subg_nodes[idx]:coor for idx, coor in enumerate(subg_pos)}     # plot
            # self.graph_plot(subg, subg_pos2draw)
            subgs.append([subg, subg_pos])

            if subg.number_of_nodes() == 1:
                # print(f"Subgraph node count: {1}")
                subgs_node2avgedge.append((1, 0))
                compintras.append([0, subg.number_of_nodes() / cluster_G.number_of_nodes()])
                continue
            subg_w = nx.get_edge_attributes(subg, 'weight')
            # print(f"subg_w: {subg_w}")
            subg_w_mean = np.mean(list(subg_w.values()))
            # print(f"subg_w_mean: {subg_w_mean}")
            subgs_node2avgedge.append((subg.number_of_nodes(), subg_w_mean))
            # sparse_sum += subg_w_mean
            compintras.append([subg_w_mean, subg.number_of_nodes() / cluster_G.number_of_nodes()])
        #     print(f"Subgraph node count: {subg.number_of_nodes()}")
        # print(f"compintras: {compintras}")

        sparse_tmps = []
        compinters = []  # Save intra-component distance and component ratio
        curclusterdict = dict()  # Record the within-cluster k-NN distance for each point in the current cluster
        records, coefs = [], []
        tmp_ds2cmps = []
        # -*- Take the component pair at the endpoints of the cutting edge -*-
        for idx, (tocutedge, cutw) in enumerate(biggeredges.items()):
            subA, subB = None, None
            for item in subgs:
                if tocutedge[0] in item[0].nodes():
                    subA, subA_pos = item[0], item[1]
                elif tocutedge[1] in item[0].nodes():
                    subB, subB_pos = item[0], item[1]

                if subA and subB: break  # Found a component pair
            # print(f"Pair {idx + 1} of components")
            # print(f"Cutting edge: {tocutedge}, length: {cutw}")
            # print(f"subA_pos shape: {subA_pos.shape}, subB_pos shape: {subB_pos.shape}")

            # Plot the component graph (only works for 2D datasets)
            subAnodes, subBnodes = list(subA), list(subB)
            subApos2draw = {subAnodes[idx]: coor for idx, coor in enumerate(subA_pos)}
            subBpos2draw = {subBnodes[idx]: coor for idx, coor in enumerate(subB_pos)}
            # self.graph_plot(subA, subApos2draw)
            # self.graph_plot(subB, subBpos2draw)

            # method0
            # -*- Within-component calculation -*-
            compinters.append([cutw, len(subAnodes) * len(subBnodes) / cluster_G.number_of_nodes() ** 2])
        # print(f"compinters: {compinters}")

        compintras = np.array(compintras)
        comp_intra = np.average(compintras[:, 0], weights=compintras[:, 1])
        # print(f"Cluster intra-component distance: {comp_intra}")
        compinters = np.array(compinters)
        comp_inter = np.average(compinters[:, 0], weights=compinters[:, 1])
        # print(f"Cluster inter-component distance: {comp_inter}")
        # sparse = np.sum([comp_intra, comp_inter])
        sparse = np.mean([comp_intra, comp_inter])
        # sparse_list = [sparse] * len(cluster)

        return sparse, contour_sample

    def cal_cluster_inter(self, clus1, contour1, clus2):
        """
        Compute the distance from the former cluster to the latter cluster.
        Note: during computation, the contour points of the former cluster are used to calculate with the latter cluster.

        :param clus1: cluster 1 (former cluster)
        :param contour1: Contour points of cluster 1
        :param clus2: cluster 2 (latter cluster)
        :return:
        """
        interk = int(np.sqrt(len(contour1)))

        def interdifference(cluster1, cluster2):
            """
            Compute the distances between point pairs in ascending order;
            The extra samples are directly ignored (thus, the inter-cluster distance depends on the smaller cluster).
            :param cluster1:
            :param cluster2:
            :return:
            """

            cluster1idxs, cluster2idxs = list(range(len(cluster1))), list(range(len(cluster2)))
            cds = cdist(cluster1, cluster2)
            vals = []
            while 0 not in cds.shape and len(vals)<interk:
                indice = np.unravel_index(np.argmin(cds), cds.shape)  # Get index of the minimum value
                minval = cds[indice] # Get the minimum value
                vals.append(minval)

                cluster1idxs.pop(indice[0])
                if 0 not in cds.shape:
                    cds = np.delete(cds, indice[0], axis=0)
                else:
                    break

                cluster2idxs.pop(indice[1])
                if 0 not in cds.shape:
                    cds = np.delete(cds, indice[1], axis=1)
                else:
                    break

            separation = np.mean(vals)
            # print(f"Mean distance of the point pairs: {np.mean(vals)}")
            return separation


        separation = interdifference(contour1, clus2)
        return separation


    @time_count
    def cal_gpvi(self):
        inter_mat = np.zeros((len(self.clusters), len(self.clusters)))  # Maintain an inter-cluster dispersion matrix
        inter_mat.fill(np.inf)
        gpvis = []
        intra_sparses, contours = [], []
        lengths = []
        used_samples_nums = []
        for idx, cur_cluster in enumerate(self.clusters):
            # intra_sparse, subgs = self.cal_cluster_intraV4(cur_cluster, self.respect_ori_indexes[idx])  # Get the intra-cluster sparseness of the current cluster
            intra_sparse, contoursamples = self.cal_cluster_intraV2(cur_cluster, self.respect_ori_indexes[idx])  # Get the intra-cluster sparseness of the current cluster
            contours.append(contoursamples)
            intra_sparses.append(intra_sparse)
            lengths.append(len(cur_cluster))
            # print(f"intra_sparses: {intra_sparses}")

            # for idx, cur_cluster in enumerate(self.clusters):
            intra_sparse = intra_sparses[idx]
            # print(f"cluster intra_sparse: {intra_sparse}")
            seps = []
            for j, clusterB in enumerate(self.clusters):
                if idx == j: continue
                separation = self.cal_cluster_inter(clus1=cur_cluster,
                                                    contour1=contoursamples,
                                                    clus2=clusterB
                                                    )
                seps.append(separation)
                inter_mat[idx][j] = separation        

        # print(f"inter_mat:{inter_mat}")
        newintermat = (inter_mat+inter_mat.T)/2
        for idx, cur_cluster in enumerate(self.clusters):

            inter_sep = min(newintermat[idx])
            # print(f"cluster inter_sep: {inter_sep}")
            intra_sparse = intra_sparses[idx]
            # print(f"cluster intra_spa: {intra_sparse}")

            gpvi = (len(self.clusters[idx]) / self.length) * (inter_sep - intra_sparse) / max(inter_sep,
                                                                                              intra_sparse)  # Get the score of the current cluster (then multiply by a weight coefficient)
            gpvis.append(gpvi)
        final_gpvi = sum(gpvis)

        return final_gpvi, gpvis

    def graph_plot(self, g, g_pos, figsize=None):
        # plot MST
        plt.figure(figsize=None)
        w = nx.get_edge_attributes(g, 'weight')
        # print(f"w: {w}")
        nx.draw(g, pos=g_pos, with_labels=True, node_color='c')
        nx.draw_networkx_edges(g, pos=g_pos, edgelist=g.edges, edge_color='orange')
        nx.draw_networkx_edge_labels(g, pos=g_pos, edge_labels=w, font_color='m')
        plt.show()

        return w


