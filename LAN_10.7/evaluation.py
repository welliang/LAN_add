# evaluation.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import CrossGraphModel, InitNodeSelectionModel, NeighborPruningModel
from data_processing import (
    read_and_split_to_individual_graph,
    make_a_dglgraph,
    readQ2GDistBook,
    get_exact_answer,
    get_topkAll_in_a_list,

)
from sklearn.metrics import roc_auc_score, accuracy_score
import networkx as nx
import heapq
import time
import jpype
import os
import numpy as np
from torch.utils.data import DataLoader
import dgl
from functools import partial

def evaluate_cross_graph_model(dataloader):
    '''
    评估跨图学习模型的性能。

    参数：
    - dataloader: 测试数据的 DataLoader。

    打印模型的测试损失和准确率。
    '''
    print("开始评估跨图学习模型...")
    # 加载模型
    model = CrossGraphModel()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('cross_graph_model.pth'))
        model.cuda()
    else:
        model.load_state_dict(torch.load('cross_graph_model.pth', map_location=torch.device('cpu')))
    model.eval()

    criterion = nn.BCELoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for bg1, bg2, labels in dataloader:
            if torch.cuda.is_available():
                bg1 = bg1.to('cuda')
                bg2 = bg2.to('cuda')
                labels = labels.float().to('cuda')
            else:
                labels = labels.float()

            outputs = model(bg1, bg2).view(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (outputs >= 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"测试损失: {avg_loss:.4f}")
    print(f"测试准确率: {accuracy * 100:.2f}%")

def getDist(q, g, distBook, estGEDBuffer, javaClass):
    '''
    计算查询图 q 与数据图 g 之间的距离（GED）。

    参数：
    - q: 查询图，NetworkX 图对象。
    - g: 数据图，NetworkX 图对象。
    - distBook: 预先计算的精确 GED 字典。
    - estGEDBuffer: 缓存的估计 GED 值字典。
    - javaClass: Java 类，用于调用 GED 计算。

    返回：
    - distance: GED 值。
    '''
    qid = q.graph.get("id")
    gid = g.graph.get("id")

    if qid == gid:
        return 0.0

    if qid in distBook and gid in distBook[qid]:
        # 使用预计算的精确 GED
        return distBook[qid][gid]
    else:
        # 使用缓存的估计 GED
        if qid in estGEDBuffer and gid in estGEDBuffer[qid]:
            return estGEDBuffer[qid][gid]
        # 调用 Java 方法估计 GED（需要确保 Java 环境已启动并正确配置）
        distance = javaClass.runApp(f"data/AIDS/g{qid}.txt", f"data/AIDS/g{gid}.txt")
        distance = distance * 2.0  # 乘以 2.0，调整为特定数据集的比例
        # 更新缓存
        estGEDBuffer.setdefault(qid, {})[gid] = distance
        estGEDBuffer.setdefault(gid, {})[qid] = distance
        return distance

def perf_measure(y_actual, y_hat):
    '''
    计算性能指标，包括 TP、FP、TN、FN。

    参数：
    - y_actual: 实际标签列表。
    - y_hat: 预测标签列表。

    返回：
    - TP: 真正例数量。
    - FP: 假正例数量。
    - TN: 真负例数量。
    - FN: 假负例数量。
    '''
    TP = FP = TN = FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    return TP, FP, TN, FN

def myloss_for_test(preds, gts):
    '''
    计算测试集上的 FNR 和 FPR。

    参数：
    - preds: 模型预测值列表。
    - gts: 实际标签列表。

    返回：
    - FN, FP, FNR, FPR: 各种指标的值。
    '''
    TP, FP, TN, FN = perf_measure(gts, preds)
    FNR = FN / (TP + FN + 1e-6)
    FPR = FP / (FP + TN + 1e-6)
    return FN, FP, FNR, FPR

def greedy_search(proxG, q, k, ep, ef, distBook, gid2gmap, estGEDBuffer, javaClass):
    '''
    使用贪婪搜索算法在邻近图中查找与查询图 q 最相似的 k 个图。

    参数：
    - proxG: 邻近图（Proximity Graph），NetworkX 图对象。
    - q: 查询图，NetworkX 图对象。
    - k: 返回的 Top-K 结果数量。
    - ep: 起始节点列表。
    - ef: 候选节点数量。
    - distBook: 预先计算的精确 GED 字典。
    - gid2gmap: 图 ID 到图对象的映射字典。
    - estGEDBuffer: 缓存的估计 GED 值字典。
    - javaClass: Java 类，用于调用 GED 计算。

    返回：
    - W: 包含 Top-K 个最近邻的堆。
    - stat: 统计信息字典，包括 DCS、hop_count 等。
    '''
    cand, stat = search_layer(proxG, q, ep, ef, distBook, gid2gmap, estGEDBuffer, javaClass)
    # 只保留 Top-K 个候选节点
    while len(cand) > k:
        heapq.heappop(cand)
    return cand, stat

def search_layer(proxG, q, ep, ef, distBook, gid2gmap, estGEDBuffer, javaClass):
    '''
    贪婪搜索的核心层，用于在邻近图中进行搜索。

    参数：
    - proxG: 邻近图（Proximity Graph），NetworkX 图对象。
    - q: 查询图，NetworkX 图对象。
    - ep: 起始节点列表。
    - ef: 候选节点数量。
    - distBook: 预先计算的精确 GED 字典。
    - gid2gmap: 图 ID 到图对象的映射字典。
    - estGEDBuffer: 缓存的估计 GED 值字典。
    - javaClass: Java 类，用于调用 GED 计算。

    返回：
    - W: 包含候选节点的堆。
    - stat: 统计信息字典，包括 DCS、hop_count 等。
    '''
    DCS = 0  # 距离计算次数
    hop_count = 0  # 跳数
    visited = set()
    C = []  # 搜索前沿，最小堆
    idx_C = 0
    W = []  # 结果集，最大堆
    idx_W = 0

    for ele in ep:
        gid = ele
        g = gid2gmap[gid]
        dist = getDist(q, g, distBook, estGEDBuffer, javaClass)
        DCS += 1
        heapq.heappush(C, (dist, idx_C, g))
        idx_C += 1
        heapq.heappush(W, (-dist, idx_W, g))
        idx_W += 1
        visited.add(gid)

    while len(C) > 0:
        c = heapq.heappop(C)
        f = min(W)
        c_dist = c[0]
        f_dist = -f[0]
        if c_dist > f_dist:
            break
        hop_count += 1
        neighbors_of_c = list(proxG.neighbors(c[2].graph.get('id')))
        for neigh_id in neighbors_of_c:
            if neigh_id not in visited:
                visited.add(neigh_id)
                neigh = gid2gmap[neigh_id]
                neigh_dist = getDist(q, neigh, distBook, estGEDBuffer, javaClass)
                DCS += 1
                if len(W) < ef or neigh_dist < -W[0][0]:
                    heapq.heappush(C, (neigh_dist, idx_C, neigh))
                    idx_C += 1
                    heapq.heappush(W, (-neigh_dist, idx_W, neigh))
                    idx_W += 1
                    if len(W) > ef:
                        heapq.heappop(W)

    stat = {
        "DCS": DCS,
        "hop_count": hop_count,
        "visited": visited
    }
    return W, stat

def evaluate_search_with_pruning(pg, queries, gid2gmap, modelMap, topk=50):
    '''
    使用邻域剪枝评估图搜索算法。

    参数：
    - pg: 邻近图（Proximity Graph），NetworkX 图对象。
    - queries: 查询图列表。
    - gid2gmap: 图 ID 到图对象的映射字典。
    - modelMap: 模型字典，用于邻域剪枝模型。
    - topk: 返回的 Top-K 结果数量。
    '''
    # 启动 Java 虚拟机，用于调用 GED 计算（请确保配置正确）
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), "-ea")
    javaClass = jpype.JClass('algorithms.GraphMatching')

    # 加载预先计算的 GED 数据
    distBook = readQ2GDistBook("data/AIDS/aids.txt")
    estGEDBuffer = {}

    avg_recall = 0.0
    avg_precision = 0.0
    avg_DCS = 0.0
    avg_hops = 0.0
    counter = 0
    start_time = time.time()

    for q in queries:
        qid = q.graph.get("id")
        print(f"查询图ID: {qid}")

        # 获取查询图的精确答案
        if qid in distBook:
            exact_ans_of_q = get_exact_answer(topk, {qid: distBook[qid]})[qid]
        else:
            print(f"查询图 {qid} 没有预计算的 GED 数据，跳过")
            continue
        print(f"查询图的精确答案: {exact_ans_of_q}")

        # 随机选择起始节点
        rand = np.random.randint(len(pg.nodes()))
        start_node = list(pg.nodes())[rand]
        start_nodes = [start_node]

        # 使用贪婪搜索算法
        cand, stat = greedy_search(pg, q, topk, start_nodes, 50, distBook, gid2gmap, estGEDBuffer, javaClass)

        pred_ans_of_q = [(ele[2].graph.get('id'), ele[0]) for ele in cand]
        pred_ans_of_q.sort(key=lambda x: x[1])
        print(f"预测的答案: {pred_ans_of_q}")
        print(f"DCS: {stat['DCS']}")

        avg_DCS += stat['DCS']
        avg_hops += stat['hop_count']

        # 计算召回率和准确率
        exact_ids = set([ele[0] for ele in exact_ans_of_q])
        pred_ids = set([ele[0] for ele in pred_ans_of_q])
        if len(exact_ids) == 0:
            recall = 0.0
        else:
            recall = len(pred_ids & exact_ids) / len(exact_ids)
        if len(pred_ids) == 0:
            precision = 0.0
        else:
            precision = len(pred_ids & exact_ids) / len(pred_ids)
        print(f"召回率: {recall}")
        print(f"准确率: {precision}")

        avg_recall += recall
        avg_precision += precision
        counter += 1
        print('-' * 50)

    end_time = time.time()
    if counter > 0:
        print(f"平均召回率: {avg_recall / counter}")
        print(f"平均准确率: {avg_precision / counter}")
        print(f"平均DCS: {avg_DCS / counter}")
        print(f"平均跳数: {avg_hops / counter}")
        print(f"平均时间 (秒): {(end_time - start_time) / counter}")
        print(f"查询总数: {counter}")
    else:
        print("没有完成任何查询评估。")

    # 关闭 Java 虚拟机
    if jpype.isJVMStarted():
        jpype.shutdownJVM()
