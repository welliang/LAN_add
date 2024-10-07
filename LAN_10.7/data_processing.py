# data_processing.py

import torch
import dgl
import networkx as nx
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import random
from functools import partial
import time

def read_and_split_to_individual_graph(fname, gsizeNoLessThan=0, gsizeLessThan=100000000, writeTo=None, prefix=None, fileformat=None, removeEdgeLabel=True, removeNodeLabel=True, graph_num=100000000):
    '''
    从文件中读取图数据，并将其拆分为单个的 NetworkX 图对象。

    参数：
    - fname: 文件名，包含所有图数据。
    - gsizeNoLessThan: 图的最小节点数量。
    - gsizeLessThan: 图的最大节点数量。
    - writeTo: 如果不为 None，则将每个图写入单独的文件。
    - prefix: 文件名前缀。
    - fileformat: 文件格式，'aids' 或 'gexf'。
    - removeEdgeLabel: 是否移除边的标签。
    - removeNodeLabel: 是否移除节点的标签。
    - graph_num: 读取的最大图数量。

    返回：
    - glist: 图列表，每个元素是一个 NetworkX 图对象。
    '''
    if writeTo is not None:
        if prefix is None:
            print("如果要将每个图写入单独的文件，需要提供文件名前缀。例如：'g' 或 'q4' 等。")
            exit(-1)
        else:
            if writeTo[-1] == '/':
                writeTo = writeTo + prefix
            else:
                writeTo = writeTo + "/" + prefix
        if fileformat is None:
            print("请指定文件格式：'aids' 或 'gexf'")
            exit(-1)

    with open(fname, 'r') as f:
        lines = f.read()

    lines2 = lines.split("t # ")
    lines3 = [g.strip().split("\n") for g in lines2]

    glist = []
    for idx in range(1, len(lines3)):
        cur_g = lines3[idx]

        gid_line = cur_g[0].strip().split(' ')
        gid = gid_line[0]
        if len(gid_line) == 4:
            glabel = gid_line[3]
            g = nx.Graph(id=gid, label=glabel)
        elif len(gid_line) == 6:
            glabel = gid_line[3]
            g = nx.Graph(id=gid, label=glabel)
        else:
            g = nx.Graph(id=gid)

        for idy in range(1, len(cur_g)):
            tmp = cur_g[idy].split(' ')
            if tmp[0] == 'v':
                if not removeNodeLabel:
                    g.add_node(tmp[1], att=tmp[2])
                else:
                    g.add_node(tmp[1], att="0")
            if tmp[0] == 'e':
                if not removeEdgeLabel:
                    g.add_edge(tmp[1], tmp[2], att=tmp[3])
                else:
                    g.add_edge(tmp[1], tmp[2], att="0")

        if g.number_of_nodes() >= gsizeNoLessThan and g.number_of_nodes() < gsizeLessThan:
            if writeTo is not None:
                if fileformat == "aids":
                    with open(writeTo + g.graph.get('id') + ".txt", "w") as f2:
                        f2.write("t # " + g.graph.get('id') + "\n")
                        if removeNodeLabel:
                            for i in range(len(g.nodes())):
                                f2.write("v " + str(i) + " 0\n")
                        else:
                            for i in range(len(g.nodes())):
                                f2.write("v " + str(i) + " " + g.nodes[str(i)].get("att") + "\n")

                        if removeEdgeLabel:
                            for e in g.edges():
                                f2.write("e " + e[0] + " " + e[1] + " 0\n")
                        else:
                            for e in g.edges():
                                f2.write("e " + e[0] + " " + e[1] + " " + g[e[0]][e[1]].get("att") + "\n")
                if fileformat == "gexf":
                    nx.write_gexf(g, writeTo + g.graph.get("id") + ".gexf")

            glist.append(g)
            if len(glist) >= graph_num:
                break

    return glist

def make_a_dglgraph(g, max_deg=20):
    '''
    将 NetworkX 图转换为 DGLGraph，并添加节点特征。

    参数：
    - g: NetworkX 图对象。
    - max_deg: 最大节点度数，用于创建 one-hot 编码。

    返回：
    - dg: DGLGraph 对象。
    '''
    ones = torch.eye(max_deg)
    edges = [[], []]
    for edge in g.edges():
        end1 = edge[0]
        end2 = edge[1]
        edges[0].append(int(end1))
        edges[1].append(int(end2))
        edges[0].append(int(end2))
        edges[1].append(int(end1))
    dg = dgl.graph((torch.tensor(edges[0]), torch.tensor(edges[1])))

    h0 = dg.in_degrees().view(-1)
    h0 = ones.index_select(0, h0.clamp(max=max_deg - 1)).float()
    dg.ndata['h'] = h0

    return dg

def read_initial_gemb(addr, emb_dim=512):
    '''
    读取初始的图嵌入。

    参数：
    - addr: 存储嵌入的文件夹地址。
    - emb_dim: 嵌入维度。

    返回：
    - gEmbMap: 图嵌入的字典，键为图ID，值为嵌入向量。
    '''
    gEmbMap = {}
    gfileList = os.listdir(addr)
    for gfile in gfileList:
        if not gfile.startswith('g') or not gfile.endswith('.txt'):
            continue
        gID = gfile[1:-4]
        with open(os.path.join(addr, gfile), 'r') as f:
            lines = f.read().strip().split('\n')
        lines = lines[1:]  # 跳过第一行
        nodeEmbList = []
        for line in lines:
            tmp = line.strip().split(' ')
            tmp2 = [float(ele) for ele in tmp[1:]]
            nodeEmbList.append(tmp2)
        nodeEmbList = torch.tensor(nodeEmbList)
        gEmb = torch.mean(nodeEmbList, 0)
        gEmbMap[gID] = gEmb
    return gEmbMap

def make_big_init_emb_tensor(gID2InitEmbMap, hdim):
    '''
    创建一个大的初始嵌入张量。

    参数：
    - gID2InitEmbMap: 图ID到嵌入的映射字典。
    - hdim: 嵌入维度。

    返回：
    - embList: 大的嵌入张量。
    - gid2posMap: 图ID到张量位置的映射字典。
    '''
    gid2posMap = {}
    embList = []
    for k, v in gID2InitEmbMap.items():
        embList.append(v)
        gid2posMap[k] = len(embList) - 1
    embList.append(torch.zeros(hdim))  # 添加一个全零向量，用于填充
    embList = torch.stack(embList)
    return embList, gid2posMap

def readQ2GDistBook(fname, validNodeIDSet=None):
    '''
    读取查询图到数据图的距离（GED）。

    参数：
    - fname: 文件名，包含距离数据。
    - validNodeIDSet: 有效的节点ID集合，如果提供，只读取这些节点的数据。

    返回：
    - distBook: 距离字典，格式为 {查询图ID: {数据图ID: 距离}}
    '''
    with open(fname, 'r') as f:
        lines = f.read().strip().split('\n')
    distBook = {}
    for line in lines:
        tmp = line.strip().split(' ')
        if validNodeIDSet is not None and tmp[1] not in validNodeIDSet:
            continue
        if tmp[0] in distBook:
            distBook[tmp[0]][tmp[1]] = float(tmp[2])
        else:
            distBook[tmp[0]] = {tmp[1]: float(tmp[2])}
    return distBook

def get_exact_answer(topk, Q2GDistBook):
    '''
    获取查询图的精确答案（Top-K）。

    参数：
    - topk: Top-K 的值。
    - Q2GDistBook: 查询图到数据图的距离字典。

    返回：
    - answer: 精确答案字典，格式为 {查询图ID: [(数据图ID, 距离), ...]}
    '''
    answer = {}
    for query in Q2GDistBook.keys():
        distToGList = list(Q2GDistBook[query].items())
        distToGList.sort(key=lambda x: x[1])
        dist_thr = -1
        if topk - 1 < len(distToGList):
            dist_thr = distToGList[topk - 1][1]
        else:
            dist_thr = 1000000.0
        a = []
        for ele in distToGList:
            if ele[1] <= dist_thr:
                a.append(ele)
            else:
                break
        answer[query] = a
    return answer

def get_topkAll_in_a_list(topk, x):
    '''
    获取距离列表中距离小于等于第K个距离的所有元素。

    参数：
    - topk: Top-K 的值。
    - x: 距离列表。

    返回：
    - res: 满足条件的元素列表。
    '''
    if topk - 1 < len(x):
        kth = x[topk - 1]
    else:
        kth = x[-1]
    res = x[:topk]
    for i in range(topk, len(x)):
        if x[i][1] == kth[1]:
            res.append(x[i])
    return res

class GINDataset(Dataset):
    '''
    GIN 数据集类，用于初始化节点选择模型的训练。

    参数：
    - name: 数据集名称。
    - database: 数据库图列表。
    - queries: 查询图ID列表。
    - exact_ans: 精确答案字典。
    - isTrain: 是否为训练集。
    '''

    def __init__(self, name, database, queries, exact_ans, isTrain=True):
        self._name = name
        self.database = database
        self.queries = queries
        self.exact_ans = exact_ans
        self.isTrain = isTrain

        self.qList = []
        self.gPosList = []
        self.ground_truth = []

        self.process()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.qList[idx], self.gPosList[idx], self.ground_truth[idx]

    def get_topkAll_in_a_list(self, topk, x):
        '''
        获取距离列表中距离小于等于第K个距离的所有元素。

        参数：
        - topk: Top-K 的值。
        - x: 距离列表。

        返回：
        - res: 满足条件的元素列表。
        '''
        kth = x[topk - 1]
        res = x[:topk]
        for i in range(topk, len(x)):
            if x[i][1] == kth[1]:
                res.append(x[i])
        return res

    def process(self):
        '''
        处理数据，生成训练所需的列表。
        '''
        for q in self.queries:
            self.qList.append(q)

            exact_ans_of_q = self.get_topkAll_in_a_list(200, self.exact_ans[q])
            exact_ans_of_q_IDSet = set()

            gt_label = []
            gPos = []
            for cur_ans in exact_ans_of_q:
                exact_ans_of_q_IDSet.add(cur_ans[0])

            for idx in range(len(self.database)):
                ele = self.database[idx]
                if ele.graph.get('id') in exact_ans_of_q_IDSet:
                    gt_label.append(1.0)
                    gPos.append(idx)
                else:
                    if self.isTrain:
                        rand = np.random.randint(10)
                        if rand > 8:
                            gt_label.append(0.0)
                            gPos.append(idx)
                    else:
                        gt_label.append(0.0)
                        gPos.append(idx)

            if len(gt_label) != len(gPos):
                print("len(gt_label) != len(gPos)")
                exit(-1)

            self.ground_truth.append(gt_label)
            self.gPosList.append(gPos)

def collate(samples):
    qids, gPosList, gtlabels = map(list, zip(*samples))
    return qids, gPosList, gtlabels

def make_big_init_emb_tensor(gID2InitEmbMap, hdim):
    '''
    创建一个大的初始嵌入张量。

    参数：
    - gID2InitEmbMap: 图ID到嵌入的映射字典。
    - hdim: 嵌入维度。

    返回：
    - embList: 大的嵌入张量。
    - gid2posMap: 图ID到张量位置的映射字典。
    '''
    gid2posMap = {}
    embList = []
    for k, v in gID2InitEmbMap.items():
        embList.append(v)
        gid2posMap[k] = len(embList) - 1
    embList.append(torch.zeros(hdim))
    embList = torch.stack(embList)
    return embList, gid2posMap

def prepare_dataloader(dataset, batch_size=32, shuffle=True):
    '''
    准备数据加载器。

    参数：
    - dataset: 数据集。
    - batch_size: 批次大小。
    - shuffle: 是否打乱数据。

    返回：
    - dataloader: 数据加载器。
    '''
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        drop_last=False,
        shuffle=shuffle
    )
    return dataloader

def read_PG(fname):
    '''
    读取邻近图（Proximity Graph）。

    参数：
    - fname: 文件名。

    返回：
    - glist: 图列表，每个元素是一个 NetworkX 图对象。
    '''
    with open(fname, 'r') as f:
        lines = f.read()

    lines2 = lines.split("t # ")
    lines3 = [g.strip().split("\n") for g in lines2]

    glist = []
    for idx in range(1, len(lines3)):
        cur_g = lines3[idx]

        gid_line = cur_g[0].strip().split(' ')
        gid = gid_line[0]
        g = nx.Graph(id=gid)

        for idy in range(1, len(cur_g)):
            tmp = cur_g[idy].split(' ')
            if tmp[0] == 'v':
                g.add_node(tmp[1], att="0")
            if tmp[0] == 'e':
                g.add_edge(tmp[1], tmp[2], ged=float(tmp[3]))

        glist.append(g)

    return glist
