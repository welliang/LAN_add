# main.py

import argparse
import torch
import os
import random
import numpy as np
import time
from data_processing import (
    read_and_split_to_individual_graph,
    make_a_dglgraph,
    read_initial_gemb,
    make_big_init_emb_tensor,
    readQ2GDistBook,
    get_exact_answer,
    GINDataset,
    prepare_dataloader,
    collate,
    read_PG
)
from models import (
    CrossGraphModel,
    InitNodeSelectionModel,
    NeighborPruningModel
)
from training import (
    train_cross_graph_model,
    train_init_node_selection_model,
    train_neighbor_pruning_model
)
from evaluation import evaluate_cross_graph_model, evaluate_search_with_pruning
import dgl
import networkx as nx  # 确保导入 networkx
from torch.utils.data import DataLoader

def set_seed(seed=42):
    '''
    设置随机种子，保证实验的可复现性。

    参数：
    - seed: 随机种子数。
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="图相似性搜索模型")

    parser.add_argument('--mode', type=str, default='train_init_node_selection',
                        choices=['train_init_node_selection'],
                        help='选择运行模式')
    parser.add_argument('--dataset', type=str, default='AIDS', help='数据集名称')
    parser.add_argument('--data_path', type=str, default='data/AIDS/', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--topk', type=int, default=50, help='Top-K 值')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 读取数据集
    print("正在读取数据集...")
    data_file = os.path.join(args.data_path, f'{args.dataset.lower()}.txt')
    glist = read_and_split_to_individual_graph(data_file)
    print(f"读取到 {len(glist)} 个图")

    # 构建图ID到图对象的映射
    gid2gmap = {g.graph.get("id"): g for g in glist}
    print("构建图ID到图对象的映射完成")

    # 构建图ID到DGLGraph的映射
    gid2dgmap = {}
    for g in glist:
        dg = make_a_dglgraph(g)
        dg = dgl.add_self_loop(dg)
        gid2dgmap[g.graph.get('id')] = dg
    print("构建图ID到DGLGraph的映射完成")

    # 读取初始图嵌入
    emb_path = os.path.join(args.data_path, 'emb')
    if os.path.exists(emb_path):
        gID2InitEmbMap = read_initial_gemb(emb_path)
        print("读取初始图嵌入完成")
    else:
        print(f"初始图嵌入文件夹 {emb_path} 不存在，请检查路径")
        return

    # 创建初始嵌入张量和映射
    emb_dim = 512  # 嵌入维度，需要根据实际情况设置
    gInitEmbBigTensor, gID2InitTensorIndexMap = make_big_init_emb_tensor(gID2InitEmbMap, emb_dim)
    if torch.cuda.is_available():
        gInitEmbBigTensor = gInitEmbBigTensor.cuda()
    print("创建初始嵌入张量和映射完成")

    # 根据模式选择运行流程
    if args.mode == 'train_init_node_selection':
        print("开始训练初始节点选择模型...")
        # 划分训练集和测试集
        num_graphs = len(glist)
        num_train = int(0.8 * num_graphs)
        database = glist[:num_train]
        queries = glist[num_train:]

        # 获取查询图ID列表
        train_queries = [g.graph.get('id') for g in queries]

        # 读取查询图到数据图的GED距离
        ged_file = os.path.join(args.data_path, f'{args.dataset.lower()}_ged.txt')
        if os.path.exists(ged_file):
            q2g_dist_book = readQ2GDistBook(ged_file)
            print("读取查询图到数据图的GED距离完成")
        else:
            print(f"GED距离文件 {ged_file} 不存在，请检查路径")
            return

        # 获取精确答案
        exact_ans = get_exact_answer(args.topk, q2g_dist_book)
        print("获取精确答案完成")

        # 创建训练数据集
        train_dataset = GINDataset(args.dataset, database, train_queries, exact_ans, isTrain=True)
        # 创建数据加载器
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate
        )
        # 创建数据库图嵌入张量
        database_emb_list = []
        for g in database:
            gid = g.graph.get('id')
            if gid in gID2InitEmbMap:
                database_emb_list.append(gID2InitEmbMap[gid])
            else:
                database_emb_list.append(torch.zeros(emb_dim))
        databaseGEmb = torch.stack(database_emb_list)
        if torch.cuda.is_available():
            databaseGEmb = databaseGEmb.cuda()
        print("创建数据库图嵌入张量完成")

        # 训练模型
        train_init_node_selection_model(dataloader, gID2InitEmbMap, gid2dgmap, databaseGEmb,
                                        epochs=args.epochs, learning_rate=args.learning_rate)
    else:
        print("未知的运行模式，请检查 --mode 参数。")

if __name__ == '__main__':
    main()
