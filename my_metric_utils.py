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

        :param clusters: 分好簇的数据
        :param label: 分好簇的标签
        :param respect_ori_indexes:
        :param length: 数据集总长度
        """
        self.clusters = clusters
        self.label = label
        self.respect_ori_indexes = respect_ori_indexes
        self.length = length

        self.wholestd = np.linalg.norm(np.std(np.concatenate(clusters, axis=0)))

    def weighted_std(self, xs, ws):
        """
        计算加权标准差
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
        计算其他组件到选择的组件对的最小距离的均值
        :param curclusterlen: 当前簇的样本数
        :param subgs: 子图集合
        :param suba: 子图1
        :param suba_pos: 子图1的samples
        :param subb: 子图2
        :param subb_pos: 子图2的samples
        :return:
        """
        lf_comp_betwn_dsts = []
        if suba and subb:
            sub_concat_cluster = np.concatenate([suba_pos, subb_pos], axis=0)
            for subgph, subgphspls in subgs:
                if subgph not in [suba, subb]:
                    tmpmin = np.min(cdist(subgphspls, sub_concat_cluster))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # 开个根号
        elif not subb:
            for subgph, subgphspls in subgs:
                if subgph not in [suba]:
                    tmpmin = np.min(cdist(subgphspls, suba_pos))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # 开个根号
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

        # 画图
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
        :return: 最小生成树:nx.classes.graph, MST的边的权值:dict
        """

        G_MST = nx.minimum_spanning_tree(G, weight='weight')
        w = nx.get_edge_attributes(G_MST, 'weight')
        # print(f"MST weight: {w}")

        """
        # 画最小生成树
        # plt.rcParams['figure.figsize'] = (8, 8)  # 设置画布大小
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
        :return: 返回跟networkx格式一样的graph
        """
        RNGdf = returnRNG.returnRNG(distance_matrix)
        print(RNGdf)
        RNG = nx.from_pandas_adjacency(RNGdf)

        # 画图
        # plt.figure(figsize=figsize)
        # nx.draw(RNG, pos=g_pos, with_labels=True, node_color='c', alpha=0.8)

        return RNG


    def cal_cluster_intraV2(self, cluster, cur_cluster_ori_idxs) :
        """
        与V7 V8相比的更新:
        组件成对时取的是切割边两端的组件
        计算类内时:
        1.组件内：通过RNG或者MST，将边取平均得到各组件内距离，再加权平均（以sub.number_of_nodes()/len(cluster)为权重）
        2.组件间：直接取连边，将这些连边们加权平均（以subA.number_of_nodes()*subB.number_of_nodes()/len(cluster)**2为权重）
        最后两者平均
        :param cluster:ndarray[n_sample, m_feature] 某个簇
        :return:
        """
        if len(cluster)==1:   # 如果数据簇只有一个样本
            # print(f"数据簇只有一个样本, shape: {cluster.shape}")
            return [0]

        cluster_G, cluster_pos, cluster_dst = self.gen_graph(cluster, cur_cluster_ori_idxs)
        cluster_MST, cluster_MST_w = self.gen_mst(G=cluster_G, pos=cluster_pos)
        # print(f"cluster_MST_w: {cluster_MST_w}")
        # print(f"这个簇的MST avg 是: {np.mean(list(cluster_MST_w.values()))}")
        # cluster_MST_nodes = cluster_MST.nodes()

        """顺便获取轮廓的点"""
        contour_node_number = np.array([x for x in cluster_MST.nodes() if cluster_MST.degree(x) == 1])
        # print(f"轮廓点数量: {len(contour_node_number)}\n轮廓点：{contour_node_number}")
        # 1号方法
        # contour_sample = cluster[np.where(cur_cluster_ori_idxs == contour_node_number[:, None])[-1]]
        # 2号方法
        # contour_sample = np.concatenate([np.expand_dims(node_pos, axis=0) for node_num,node_pos in cluster_pos.items() if node_num in contour_node_number], axis=0)
        # 3号方法
        contour_sample = []
        for node_num in contour_node_number:
            if node_num in cluster_pos:
                contour_sample.append(np.expand_dims(cluster_pos[node_num], axis=0))
        contour_sample = np.concatenate(contour_sample, axis=0)
        # print(f"length of contour_node_number: {len(contour_node_number)}, length of contour_sample: {len(contour_sample)}")

        """找到MST的最大边, 并将该边顶点放入轮廓， 然后再对最大边进行切割"""
        sorted_w_of_MST = sorted([item for item in cluster_MST_w.items()], key=lambda x: x[1], reverse=True)
        # print(f"按边长从大到小排序后: {sorted_w_of_MST}")
        max_edge = sorted_w_of_MST[0][0]
        # print(f"对应的最大边为: {max_edge}, 边长是: {sorted_w_of_MST[0][1]}")

        # 取大于avg+3*标准差的那些边
        biggeredges = {max_edge:sorted_w_of_MST[0][1]}
        edge_avg, edge_std = np.mean(list(cluster_MST_w.values())), np.std(list(cluster_MST_w.values()))
        for edge, w in cluster_MST_w.items():
            if w > edge_avg + 3 * edge_std:   # 只看3*sigma
                biggeredges[edge] = w

        # for time in range(3, 0, -1):          # 1,2,3个sigma都尝试一次
        #     for edge, w in cluster_MST_w.items():
        #         if w > edge_avg + time * edge_std:
        #             biggeredges[edge] = w
        #     if biggeredges:
        #         break
        # print(f"biggeredges: {biggeredges}")


        # 切掉MST中这些大于avg+3*标准差的那些边
        for tocutedge, w in biggeredges.items():
            # print(tocutedge)
            cluster_MST.remove_edge(*tocutedge)

        # 获取切割完成后的所得子图
        subgs_node2avgedge = []
        subgs = []
        compintras = []  # 记录组件内的距离 与 组件的比例
        sparse_sum = 0
        for connect_item in nx.connected_components(cluster_MST):
            subg = cluster_MST.subgraph(connect_item)
            # print(f"subg: {subg.adj}")
            subg_nodes = np.array([x for x in subg.nodes()])  # 或者直接list(subg)
            subg_pos = cluster[np.where(cur_cluster_ori_idxs == subg_nodes[:, None])[-1]]   # 其实是samples
            # subg_pos2draw = {subg_nodes[idx]:coor for idx, coor in enumerate(subg_pos)}     # 这个才是用于画画的
            # self.graph_plot(subg, subg_pos2draw)
            subgs.append([subg, subg_pos])

            if subg.number_of_nodes() == 1:
                # print(f"子图的节点数: {1}")
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
        #     print(f"子图的节点数: {subg.number_of_nodes()}")
        # print(f"compintras: {compintras}")

        sparse_tmps = []
        compinters = []     # 记录组件间的距离 与 组件对的比例
        curclusterdict = dict()     # 记录当前簇的各点的类内knn距离
        records, coefs = [], []
        tmp_ds2cmps = []
        # -*- 取切割边两端的组件对 -*-
        for idx, (tocutedge, cutw) in enumerate(biggeredges.items()):
            subA, subB = None, None
            for item in subgs:
                if tocutedge[0] in item[0].nodes():
                    subA, subA_pos = item[0], item[1]
                elif tocutedge[1] in item[0].nodes():
                    subB, subB_pos = item[0], item[1]

                if subA and subB: break    # 表示找到成对的组件了
            # print(f"第{idx + 1}对组件")
            # print(f"切割边是: {tocutedge}，长度是：{cutw}")
            # print(f"subA_pos shape: {subA_pos.shape}, subB_pos shape: {subB_pos.shape}")


            # 查看组件图(注意只能用于2维数据集的时候才能画图)
            subAnodes, subBnodes = list(subA), list(subB)
            subApos2draw = {subAnodes[idx]: coor for idx, coor in enumerate(subA_pos)}
            subBpos2draw = {subBnodes[idx]: coor for idx, coor in enumerate(subB_pos)}
            # self.graph_plot(subA, subApos2draw)
            # self.graph_plot(subB, subBpos2draw)

            # 法0
            # -*- 组件内的计算 -*-
            compinters.append([cutw, len(subAnodes)*len(subBnodes)/cluster_G.number_of_nodes()**2])
        # print(f"compinters: {compinters}")

        compintras = np.array(compintras)
        comp_intra = np.average(compintras[:, 0], weights=compintras[:, 1])
        # print(f"该簇的组件内距离：{comp_intra}")
        compinters = np.array(compinters)
        comp_inter = np.average(compinters[:, 0], weights=compinters[:, 1])
        # print(f"该簇的组件间距离：{comp_inter}")
        # sparse = np.sum([comp_intra, comp_inter])
        sparse = np.mean([comp_intra, comp_inter])
        sparse_list = [sparse]*len(cluster)

        return sparse_list


    @time_count
    def cal_gpvim(self):
        """
        计算
        :return:
        """
        inter_mat = np.zeros((len(self.clusters), len(self.clusters)))  # 维护一个类间离散度矩阵
        inter_mat.fill(np.inf)
        gpvis = []
        # intra_sparses, comps = [], []
        lengths = []
        all_point_dis = []
        all_s = []
        for idx, cur_cluster in enumerate(self.clusters):
            intra_sparses = self.cal_cluster_intraV2(cur_cluster, self.respect_ori_indexes[idx])  # 获取当前类中各点类内knn距离
            # intra_sparses.append(intra_sparse)
            assert len(cur_cluster) == len(intra_sparses)
            # print(f"MLI当前簇各点intra_sparses: {intra_sparses}")
            lengths.append(len(cur_cluster))
            pointdis = []
            tmp_all_s = []
            for i, instance in enumerate(cur_cluster):
                bi = float("inf")
                for j, clusterB in enumerate(self.clusters):
                    if idx == j: continue
                    intercdst = cdist(np.expand_dims(instance, axis=0), clusterB)
                    """计算当前点的全局类间距离"""
                    # tmp_inter_dis = np.mean(intercdst)
                    """计算当前点的knn类间距离"""
                    interk = int(np.sqrt(len(clusterB)))
                    vals2B, idxs2B = topk_(-intercdst, K=interk, axis=1)
                    vals2B = -vals2B  # 取反
                    tmp_inter_dis = np.sum(vals2B, axis=1).item() / (interk - 1) if interk>1 else np.mean(vals2B)
                    if tmp_inter_dis < bi:
                        bi = tmp_inter_dis
                # print(f"MTL当前点的类内距离：{intra_sparses[i]}")
                # print(f"MTL当前点的类间距离：{bi}")
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
        sc = np.mean(new_all_s)  # 直接平均
        # print(f"new_all_s: {new_all_s}")
        # print(f"sc: {sc}")

        # 检查数量是否相等
        for i, clu in enumerate(self.clusters):
            # print(f"len of clu: {len(clu)}")
            # print(f"len of sc: {len(all_s[i])}")
            assert len(clu) == len(all_s[i])



        return sc, (all_s, all_point_dis)

    def graph_plot(self, g, g_pos, figsize=None):
        # 画最小生成树
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

        :param clusters: 分好簇的数据
        :param label: 分好簇的标签
        :param respect_ori_indexes:
        :param length: 数据集总长度
        """
        self.clusters = clusters
        self.label = label
        self.respect_ori_indexes = respect_ori_indexes
        self.length = length

        self.wholestd = np.linalg.norm(np.std(np.concatenate(clusters, axis=0)))

    def weighted_std(self, xs, ws):
        """
        计算加权标准差
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
        计算其他组件到选择的组件对的最小距离的均值
        :param curclusterlen: 当前簇的样本数
        :param subgs: 子图集合
        :param suba: 子图1
        :param suba_pos: 子图1的samples
        :param subb: 子图2
        :param subb_pos: 子图2的samples
        :return:
        """
        lf_comp_betwn_dsts = []
        if suba and subb:
            sub_concat_cluster = np.concatenate([suba_pos, subb_pos], axis=0)
            for subgph, subgphspls in subgs:
                if subgph not in [suba, subb]:
                    tmpmin = np.min(cdist(subgphspls, sub_concat_cluster))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # 开个根号
        elif not subb:
            for subgph, subgphspls in subgs:
                if subgph not in [suba]:
                    tmpmin = np.min(cdist(subgphspls, suba_pos))
                    lf_comp_betwn_dsts.append([tmpmin, np.sqrt(subgph.number_of_nodes() / curclusterlen)])  # 开个根号
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

        # 画图
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
        :return: 最小生成树:nx.classes.graph, MST的边的权值:dict
        """

        G_MST = nx.minimum_spanning_tree(G, weight='weight')
        w = nx.get_edge_attributes(G_MST, 'weight')
        # print(f"MST weight: {w}")

        """
        # 画最小生成树
        # plt.rcParams['figure.figsize'] = (8, 8)  # 设置画布大小
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
        :return: 返回跟networkx格式一样的graph
        """
        RNGdf = returnRNG.returnRNG(distance_matrix)
        print(RNGdf)
        RNG = nx.from_pandas_adjacency(RNGdf)

        # 画图
        # plt.figure(figsize=figsize)
        # nx.draw(RNG, pos=g_pos, with_labels=True, node_color='c', alpha=0.8)

        return RNG


    def cal_cluster_intraV2(self, cluster, cur_cluster_ori_idxs):
        """
        与V7 V8相比的更新:
        组件成对时取的是切割边两端的组件
        计算类内时:
        1.组件内：通过RNG或者MST，将边取平均得到各组件内距离，再加权平均（以sub.number_of_nodes()/len(cluster)为权重）
        2.组件间：直接取连边，将这些连边们加权平均（以subA.number_of_nodes()*subB.number_of_nodes()/len(cluster)**2为权重）
        最后两者平均
        :param cluster:ndarray[n_sample, m_feature] 某个簇
        :return:
        """
        if len(cluster)==1:   # 如果数据簇只有一个样本
            # print(f"数据簇只有一个样本, shape: {cluster.shape}")
            return 0, cluster

        cluster_G, cluster_pos, cluster_dst = self.gen_graph(cluster, cur_cluster_ori_idxs)
        cluster_MST, cluster_MST_w = self.gen_mst(G=cluster_G, pos=cluster_pos)
        # print(f"cluster_MST_w: {cluster_MST_w}")
        # print(f"这个簇的MST avg 是: {np.mean(list(cluster_MST_w.values()))}")
        # cluster_MST_nodes = cluster_MST.nodes()

        """顺便获取轮廓的点"""
        contour_node_number = np.array([x for x in cluster_MST.nodes() if cluster_MST.degree(x) == 1])
        # print(f"轮廓点数量: {len(contour_node_number)}\n轮廓点：{contour_node_number}")
        # 1号方法
        contour_sample = cluster[np.where(cur_cluster_ori_idxs == contour_node_number[:, None])[-1]]
        # 2号方法
        # contour_sample = np.concatenate([np.expand_dims(node_pos, axis=0) for node_num,node_pos in cluster_pos.items() if node_num in contour_node_number], axis=0)
        # 3号方法
        # contour_sample = []
        # for node_num in contour_node_number:
        #     if node_num in cluster_pos:
        #         contour_sample.append(np.expand_dims(cluster_pos[node_num], axis=0))
        # contour_sample = np.concatenate(contour_sample, axis=0)
        # print(f"length of contour_node_number: {len(contour_node_number)}, "
        #       f"length of contour_sample: {len(contour_sample)}")

        """找到MST的最大边, 并将该边顶点放入轮廓， 然后再对最大边进行切割"""
        sorted_w_of_MST = sorted([item for item in cluster_MST_w.items()], key=lambda x: x[1], reverse=True)
        # print(f"按边长从大到小排序后: {sorted_w_of_MST}")
        max_edge = sorted_w_of_MST[0][0]
        # print(f"对应的最大边为: {max_edge}, 边长是: {sorted_w_of_MST[0][1]}")

        # 取大于avg+3*标准差的那些边
        biggeredges = {max_edge: sorted_w_of_MST[0][1]}
        edge_avg, edge_std = np.mean(list(cluster_MST_w.values())), np.std(list(cluster_MST_w.values()))
        for edge, w in cluster_MST_w.items():
            if w > edge_avg + 3 * edge_std:  # 只看3*sigma
                biggeredges[edge] = w

        # for time in range(3, 0, -1):          # 1,2,3个sigma都尝试一次
        #     for edge, w in cluster_MST_w.items():
        #         if w > edge_avg + time * edge_std:
        #             biggeredges[edge] = w
        #     if biggeredges:
        #         break
        # print(f"biggeredges: {biggeredges}")

        # 切掉MST中这些大于avg+3*标准差的那些边
        for tocutedge, w in biggeredges.items():
            # print(tocutedge)
            cluster_MST.remove_edge(*tocutedge)

        # 获取切割完成后的所得子图
        subgs_node2avgedge = []
        subgs = []
        compintras = []  # 记录组件内的距离 与 组件的比例
        sparse_sum = 0
        for connect_item in nx.connected_components(cluster_MST):
            subg = cluster_MST.subgraph(connect_item)
            # print(f"subg: {subg.adj}")
            subg_nodes = np.array([x for x in subg.nodes()])  # 或者直接list(subg)
            subg_pos = cluster[np.where(cur_cluster_ori_idxs == subg_nodes[:, None])[-1]]  # 其实是samples
            # subg_pos2draw = {subg_nodes[idx]:coor for idx, coor in enumerate(subg_pos)}     # 这个才是用于画画的
            # self.graph_plot(subg, subg_pos2draw)
            subgs.append([subg, subg_pos])

            if subg.number_of_nodes() == 1:
                # print(f"子图的节点数: {1}")
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
        #     print(f"子图的节点数: {subg.number_of_nodes()}")
        # print(f"compintras: {compintras}")

        sparse_tmps = []
        compinters = []  # 记录组件间的距离 与 组件对的比例
        curclusterdict = dict()  # 记录当前簇的各点的类内knn距离
        records, coefs = [], []
        tmp_ds2cmps = []
        # -*- 取切割边两端的组件对 -*-
        for idx, (tocutedge, cutw) in enumerate(biggeredges.items()):
            subA, subB = None, None
            for item in subgs:
                if tocutedge[0] in item[0].nodes():
                    subA, subA_pos = item[0], item[1]
                elif tocutedge[1] in item[0].nodes():
                    subB, subB_pos = item[0], item[1]

                if subA and subB: break  # 表示找到成对的组件了
            # print(f"第{idx + 1}对组件")
            # print(f"切割边是: {tocutedge}，长度是：{cutw}")
            # print(f"subA_pos shape: {subA_pos.shape}, subB_pos shape: {subB_pos.shape}")

            # 查看组件图(注意只能用于2维数据集的时候才能画图)
            subAnodes, subBnodes = list(subA), list(subB)
            subApos2draw = {subAnodes[idx]: coor for idx, coor in enumerate(subA_pos)}
            subBpos2draw = {subBnodes[idx]: coor for idx, coor in enumerate(subB_pos)}
            # self.graph_plot(subA, subApos2draw)
            # self.graph_plot(subB, subBpos2draw)

            # 法0
            # -*- 组件内的计算 -*-
            compinters.append([cutw, len(subAnodes) * len(subBnodes) / cluster_G.number_of_nodes() ** 2])
        # print(f"compinters: {compinters}")

        compintras = np.array(compintras)
        comp_intra = np.average(compintras[:, 0], weights=compintras[:, 1])
        # print(f"该簇的组件内距离：{comp_intra}")
        compinters = np.array(compinters)
        comp_inter = np.average(compinters[:, 0], weights=compinters[:, 1])
        # print(f"该簇的组件间距离：{comp_inter}")
        # sparse = np.sum([comp_intra, comp_inter])
        sparse = np.mean([comp_intra, comp_inter])
        # sparse_list = [sparse] * len(cluster)

        return sparse, contour_sample

    def cal_cluster_inter(self, clus1, contour1, clus2):
        """
        计算前者类到后者类的距离
        注意计算的时候是用的前者类的轮廓点来跟后者类计算
        :param clus1: 簇1(前者类)
        :param contour1: 簇1的轮廓点
        :param clus2: 簇2(后者类)
        :return:
        """
        interk = int(np.sqrt(len(contour1)))

        def interdifference(cluster1, cluster2):
            """
            从小到大计算点对之间的距离;
            多出来的那部分样本直接忽略(这样的话, 也就是说类间距离取决于小簇)
            :param cluster1:
            :param cluster2:
            :return:
            """

            cluster1idxs, cluster2idxs = list(range(len(cluster1))), list(range(len(cluster2)))
            cds = cdist(cluster1, cluster2)
            vals = []
            while 0 not in cds.shape and len(vals)<interk:
                indice = np.unravel_index(np.argmin(cds), cds.shape)  # 获取最小值的位置
                minval = cds[indice]  # 获取最小值
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
            # print(f"点对部分的均值距离: {np.mean(vals)}")
            return separation


        separation = interdifference(contour1, clus2)
        return separation


    @time_count
    def cal_gpvi(self):
        inter_mat = np.zeros((len(self.clusters), len(self.clusters)))  # 维护一个类间离散度矩阵
        inter_mat.fill(np.inf)
        gpvis = []
        intra_sparses, contours = [], []
        lengths = []
        used_samples_nums = []
        for idx, cur_cluster in enumerate(self.clusters):
            # intra_sparse, subgs = self.cal_cluster_intraV4(cur_cluster, self.respect_ori_indexes[idx])  # 获取当前类的类内稀疏度
            intra_sparse, contoursamples = self.cal_cluster_intraV2(cur_cluster, self.respect_ori_indexes[idx])  # 获取当前类的类内稀疏度
            contours.append(contoursamples)
            intra_sparses.append(intra_sparse)
            lengths.append(len(cur_cluster))
            # print(f"intra_sparses: {intra_sparses}")

            # for idx, cur_cluster in enumerate(self.clusters):
            intra_sparse = intra_sparses[idx]
            # print(f"当前簇intra_sparse: {intra_sparse}")
            seps = []
            for j, clusterB in enumerate(self.clusters):
                if idx == j: continue
                separation = self.cal_cluster_inter(clus1=cur_cluster,
                                                    contour1=contoursamples,
                                                    clus2=clusterB
                                                    )
                seps.append(separation)
                inter_mat[idx][j] = separation        # A类到B类的距离与B类到A类的距离有可能些许差异

        # print(f"inter_mat:{inter_mat}")
        newintermat = (inter_mat+inter_mat.T)/2
        for idx, cur_cluster in enumerate(self.clusters):

            inter_sep = min(newintermat[idx])
            # print(f"当前这个簇inter_sep: {inter_sep}")
            intra_sparse = intra_sparses[idx]
            # print(f"当前这个簇intra_spa: {intra_sparse}")

            gpvi = (len(self.clusters[idx]) / self.length) * (inter_sep - intra_sparse) / max(inter_sep,
                                                                                              intra_sparse)  # 得到当前簇的分数(再乘上一个权重系数)
            gpvis.append(gpvi)
        final_gpvi = sum(gpvis)

        return final_gpvi, gpvis

    def graph_plot(self, g, g_pos, figsize=None):
        # 画最小生成树
        plt.figure(figsize=None)
        w = nx.get_edge_attributes(g, 'weight')
        # print(f"w: {w}")
        nx.draw(g, pos=g_pos, with_labels=True, node_color='c')
        nx.draw_networkx_edges(g, pos=g_pos, edgelist=g.edges, edge_color='orange')
        nx.draw_networkx_edge_labels(g, pos=g_pos, edge_labels=w, font_color='m')
        plt.show()

        return w


