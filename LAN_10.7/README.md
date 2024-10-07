图相似性搜索模型
该项目实现了一个用于图相似性搜索的模型，包含以下主要功能：

数据处理：读取和处理图数据，生成用于模型训练和评估的数据集。
模型定义：定义了三个主要模型，用于跨图学习、初始节点选择和邻域剪枝。
模型训练：训练上述模型，包括定义损失函数、优化器和训练过程。
模型评估：评估模型的性能，包括计算准确率、召回率等指标。
目录结构
data_processing.py：数据处理相关的函数和类。
models.py：定义了用于图相似性搜索的模型。
training.py：包含模型的训练函数。
evaluation.py：包含模型的评估函数。
main.py：主程序，用于运行不同的模式（训练、评估等）。
环境依赖
Python 3.x
PyTorch
DGL (Deep Graph Library)
NetworkX
NumPy
JPype1（用于与Java交互，计算图编辑距离）
其他依赖库请参考 requirements.txt
安装依赖
bash
复制代码
pip install -r requirements.txt
运行方式
训练初始节点选择模型
bash
复制代码
python main.py --mode train_init_node_selection
训练跨图学习模型
bash
复制代码
python main.py --mode train_cross_graph
训练邻域剪枝模型
bash
复制代码
python main.py --mode train_neighbor_pruning
评估模型
bash
复制代码
python main.py --mode evaluate_cross_graph
python main.py --mode evaluate_search_with_pruning
参数说明
--mode：运行模式，支持 train_init_node_selection、train_cross_graph、train_neighbor_pruning、evaluate_cross_graph、evaluate_search_with_pruning。
--dataset：数据集名称，默认是 AIDS。
--data_path：数据集路径，默认是 data/AIDS/。
--epochs：训练轮数，默认是 10。
--learning_rate：学习率，默认是 0.001。
--batch_size：批次大小，默认是 32。
--topk：Top-K 值，默认是 50。
--seed：随机种子，默认是 42。
文件功能说明
data_processing.py
包含了数据处理相关的函数和类，包括：

read_and_split_to_individual_graph：读取图数据并拆分为单个的 NetworkX 图对象。
make_a_dglgraph：将 NetworkX 图转换为 DGLGraph，并添加节点特征。
read_initial_gemb：读取初始的图嵌入。
make_big_init_emb_tensor：创建一个大的初始嵌入张量。
readQ2GDistBook：读取查询图到数据图的距离（GED）。
get_exact_answer：获取查询图的精确答案（Top-K）。
get_topkAll_in_a_list：获取距离列表中距离小于等于第 K 个距离的所有元素。
GINDataset：用于初始化节点选择模型的训练数据集类。
collate：用于数据加载器的 collate_fn 函数。
prepare_dataloader：准备数据加载器。
read_PG：读取邻近图（Proximity Graph）。
models.py
定义了用于图相似性搜索的模型，包括：

CrossGraphModel：跨图学习模型，用于计算两个图之间的相似性。
InitNodeSelectionModel：初始节点选择模型。
NeighborPruningModel：邻域剪枝模型。
training.py
包含模型的训练函数，包括：

weighted_binary_cross_entropy：自定义的加权二元交叉熵损失函数。
train_cross_graph_model：训练跨图学习模型。
train_init_node_selection_model：训练初始节点选择模型。
train_neighbor_pruning_model：训练邻域剪枝模型。
evaluation.py
包含模型的评估函数，包括：

evaluate_cross_graph_model：评估跨图学习模型的性能。
evaluate_search_with_pruning：使用邻域剪枝评估图搜索算法。
getDist：计算查询图与数据图之间的 GED 距离。
greedy_search：使用贪婪搜索算法在邻近图中查找最相似的图。
search_layer：贪婪搜索的核心层。
perf_measure：计算性能指标（TP、FP、TN、FN）。
myloss_for_test：计算测试集上的 FNR 和 FPR。
main.py
主程序，用于运行不同的模式（训练、评估等）。根据命令行参数，执行相应的训练或评估流程。

