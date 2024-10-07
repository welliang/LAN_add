# training.py
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from models import CrossGraphModel, InitNodeSelectionModel, NeighborPruningModel
from data_processing import make_big_init_emb_tensor, prepare_dataloader
from torch.utils.data import DataLoader
from functools import partial
import time
import numpy as np

def weighted_binary_cross_entropy(output, target, weights=None):
    '''
    自定义的加权二元交叉熵损失函数。

    参数：
    - output: 模型的输出。
    - target: 真实标签。
    - weights: 权重列表。

    返回：
    - loss: 损失值。
    '''
    output = torch.clamp(output, min=1e-6, max=1 - 1e-6)
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))

def train_cross_graph_model(dataloader, epochs=10, learning_rate=0.001):
    '''
    训练跨图学习模型。

    参数：
    - dataloader: 数据加载器。
    - epochs: 训练轮数。
    - learning_rate: 学习率。
    '''
    model = CrossGraphModel()
    model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        total_loss = 0
        for bg1, bg2, labels in dataloader:
            bg1 = bg1.to('cuda')
            bg2 = bg2.to('cuda')
            labels = labels.float().to('cuda')
            optimizer.zero_grad()
            preds = model(bg1, bg2).view(-1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"第 {epoch + 1} 轮，损失: {avg_loss:.4f}")
    torch.save(model.state_dict(), 'cross_graph_model.pth')

def train_init_node_selection_model(train_dataset, gID2InitEmbMap, gid2dgmap, databaseGEmb, epochs=10, learning_rate=0.001):
    '''
    训练初始节点选择模型。

    参数：
    - train_dataset: 训练数据集。
    - gID2InitEmbMap: 图初始嵌入映射。
    - gid2dgmap: 图ID到DGLGraph的映射。
    - databaseGEmb: 数据库图嵌入张量。
    - epochs: 训练轮数。
    - learning_rate: 学习率。
    '''
    model = InitNodeSelectionModel(gID2InitEmbMap, gid2dgmap, databaseGEmb)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    dataloader = prepare_dataloader(train_dataset, batch_size=1, shuffle=True)
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for qids, gPosList, gtlabels in dataloader:
            preds = model(qids, gPosList)
            # 将 preds 和 gtlabels 展平为一维张量
            preds_tensor = torch.cat(preds).cuda()
            gtlabels_tensor = torch.tensor(gtlabels[0], dtype=torch.float32).cuda()
            loss = weighted_binary_cross_entropy(preds_tensor, gtlabels_tensor, weights=[1.0, 10.0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"第 {epoch + 1} 轮，损失: {loss.item():.4f}，耗时: {time.time() - start_time:.2f}s")
    torch.save(model.state_dict(), 'init_node_selection_model.pth')

def train_neighbor_pruning_model(train_dataset, gInitEmbMap, gid2dgmap, hdim=512, outputNum=10, epochs=10, learning_rate=0.001):
    '''
    训练邻域剪枝模型。

    参数：
    - train_dataset: 训练数据集。
    - gInitEmbMap: 图初始嵌入映射。
    - gid2dgmap: 图ID到DGLGraph的映射。
    - hdim: 嵌入维度。
    - outputNum: 输出数量。
    - epochs: 训练轮数。
    - learning_rate: 学习率。
    '''
    # 创建初始嵌入张量
    gInitEmbBigTensor, gID2InitTensorIndexMap = make_big_init_emb_tensor(gInitEmbMap, hdim)
    gInitEmbBigTensor = gInitEmbBigTensor.cuda()

    # 定义数据加载器的 collate_fn
    def my_collate_fn(samples):
        qEmbs, pgNodes, neighIndexLists, gts, mask_of_1_list, mask_of_0_list, classWeightList = map(list, zip(*samples))
        neighIndexLists = torch.tensor(neighIndexLists).view(1, -1).squeeze()
        neighInitEmbs = torch.index_select(gInitEmbBigTensor, 0, neighIndexLists)
        return dgl.batch(qEmbs), torch.stack(pgNodes), neighInitEmbs, torch.tensor(gts), torch.stack(mask_of_1_list), torch.stack(mask_of_0_list), torch.tensor(classWeightList)

    dataloader = DataLoader(
        train_dataset,
        batch_size=1000,
        collate_fn=my_collate_fn,
        num_workers=6,
        drop_last=False,
        shuffle=True
    )

    model = NeighborPruningModel(hdim=hdim, outputNum=outputNum)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for qList, pgNodes, neighEmbList, gts, mask_of_1_list, mask_of_0_list, classWeightList in dataloader:
            qList = qList.to('cuda')
            pgNodes = pgNodes.to('cuda')
            neighEmbList = neighEmbList.to('cuda')
            gts = gts.float().to('cuda')
            preds, _ = model(qList, pgNodes, neighEmbList, classWeightList)
            loss = weighted_binary_cross_entropy(preds, gts, weights=[10.0, 1.0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"第 {epoch + 1} 轮，损失: {loss.item():.4f}，耗时: {time.time() - start_time:.2f}s")
    torch.save(model.state_dict(), 'neighbor_pruning_model.pth')
