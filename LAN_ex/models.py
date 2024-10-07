# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv

class CrossGraphModel(nn.Module):
    '''
    跨图学习模型，用于计算两个图之间的相似性。

    参数：
    - input_dim: 输入特征维度。
    - hidden_dim: 隐藏层维度。
    '''
    def __init__(self, input_dim=20, hidden_dim=512):
        super(CrossGraphModel, self).__init__()

        self.hdim = hidden_dim

        # 初始化节点特征的全连接层
        self.fc_init_node = nn.Linear(input_dim, self.hdim)

        # GIN卷积层
        self.conv1 = GINConv(nn.Linear(self.hdim, self.hdim), 'mean')
        self.conv2 = GINConv(nn.Linear(self.hdim, self.hdim), 'mean')

        # 批归一化层
        self.gnn_bn1 = nn.BatchNorm1d(self.hdim)
        self.gnn_bn2 = nn.BatchNorm1d(self.hdim)

        # 全连接层用于计算输出
        self.fc = nn.Linear(self.hdim * 2, 256)
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, g1, g2):
        # 对图1的节点特征进行初始化和卷积
        h0_g1 = self.fc_init_node(g1.ndata['h'])
        h1_g1 = self.relu(self.gnn_bn1(self.conv1(g1, h0_g1)))
        h2_g1 = self.relu(self.gnn_bn2(self.conv2(g1, h1_g1)))
        g1_emb = dgl.mean_nodes(g1, h2_g1)

        # 对图2的节点特征进行初始化和卷积
        h0_g2 = self.fc_init_node(g2.ndata['h'])
        h1_g2 = self.relu(self.gnn_bn1(self.conv1(g2, h0_g2)))
        h2_g2 = self.relu(self.gnn_bn2(self.conv2(g2, h1_g2)))
        g2_emb = dgl.mean_nodes(g2, h2_g2)

        # 合并两个图的嵌入并通过全连接层
        x = torch.cat([g1_emb, g2_emb], dim=1)
        x = self.relu(self.fc(x))
        out = torch.sigmoid(self.fc2(x))
        return out

class InitNodeSelectionModel(nn.Module):
    '''
    初始节点选择模型。

    参数：
    - gID2InitEmbMap: 图ID到初始嵌入的映射字典。
    - gid2dgMap: 图ID到DGLGraph的映射字典。
    - allDBGEmb: 数据库中所有图的嵌入张量。
    '''
    def __init__(self, gID2InitEmbMap, gid2dgMap, allDBGEmb):
        super(InitNodeSelectionModel, self).__init__()

        self.hdim = 1024
        self.gID2InitEmbMap = gID2InitEmbMap
        self.gid2dgMap = gid2dgMap
        self.allDBGEmb = allDBGEmb

        # GIN卷积层
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')

        # 批归一化层
        self.gnn_bn = nn.BatchNorm1d(self.hdim)
        self.gnn_bn2 = nn.BatchNorm1d(self.hdim)

        # 全连接层
        self.fc_init = nn.Linear(20, self.hdim)
        self.fc = nn.Linear(self.hdim * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, qids, gPosList):
        preds = []

        for idx in range(len(qids)):
            qid = qids[idx]
            gPos = gPosList[idx]

            # 获取查询图的DGLGraph
            dg_of_q = self.gid2dgMap[qid]

            # 对查询图进行卷积和嵌入
            dg_of_q.ndata['h2'] = self.fc_init(dg_of_q.ndata['h'])
            dg_of_q.ndata['h2'] = self.relu(self.gnn_bn(self.conv1_for_g(dg_of_q, dg_of_q.ndata['h2'])))
            dg_of_q.ndata['h2'] = self.relu(self.gnn_bn2(self.conv2_for_g(dg_of_q, dg_of_q.ndata['h2'])))
            qemb = dgl.mean_nodes(dg_of_q, 'h2').squeeze()

            # 获取数据库中对应的图嵌入
            gEmbList = self.allDBGEmb.index_select(0, torch.tensor(gPos).cuda())
            qemb = qemb.repeat(gEmbList.shape[0], 1)

            H = torch.cat([qemb, gEmbList], 1)
            H2 = self.relu(self.bn(self.fc(H)))
            probs = torch.sigmoid(self.fc2(H2)).view(-1)
            preds.append(probs)

        return preds

class NeighborPruningModel(nn.Module):
    '''
    邻域剪枝模型。

    参数：
    - hdim: 嵌入维度。
    - outputNum: 输出数量。
    '''
    def __init__(self, hdim=512, outputNum=10):
        super(NeighborPruningModel, self).__init__()

        self.hdim = hdim
        self.outputNum = outputNum

        self.relu = nn.ReLU(inplace=True)

        self.fc_init = nn.Linear(20, self.hdim)
        self.conv1_for_g = GINConv(None, 'mean')
        self.conv2_for_g = GINConv(None, 'mean')

        self.gnn_bn = nn.BatchNorm1d(self.hdim)
        self.gnn_bn2 = nn.BatchNorm1d(self.hdim)

        self.fc = nn.Linear(self.hdim * 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.bn = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.dp = nn.Dropout(0.5)

    def forward(self, qList, pgNodeEmbList, neighEmbList, classWeightList):
        batch_size = len(pgNodeEmbList)
        number_of_outputs = self.outputNum

        # 对查询图进行卷积和嵌入
        qList.ndata['h'] = self.fc_init(qList.ndata['h'])
        qList.ndata['h'] = self.relu(self.gnn_bn(self.conv1_for_g(qList, qList.ndata['h'])))
        qList.ndata['h'] = self.relu(self.gnn_bn2(self.conv2_for_g(qList, qList.ndata['h'])))
        qemb = dgl.mean_nodes(qList, 'h')

        a = torch.cat([qemb, pgNodeEmbList], 1)
        a = a.repeat(1, number_of_outputs).view(-1, self.hdim * 2)

        b = torch.cat([a, neighEmbList], 1)

        H = self.relu(self.bn(self.fc(b)))
        H2 = self.relu(self.bn2(self.fc2(H)))
        H3 = self.relu(self.bn3(self.fc3(H2)))
        preds = torch.sigmoid(self.fc4(H3))

        preds = preds.view(batch_size, number_of_outputs)

        return preds, H3
