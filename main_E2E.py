# main.py
import torch
import os
import numpy as np
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges
from config import Config
from models import EndToEndProtoModel_E
from save_experiment_data import save_experiment_data
from trainer import train_node_classification_distill,train_node_classification_distill_start

from utils import kmeans, FeatureProjection
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from evaluation import *

import torch
import numpy as np
from collections import defaultdict


def split_stratified_mask(labels, train_ratio=0.2, val_ratio=0.2, test_ratio=0.6, seed=42):
    """
    按类别分层随机划分训练/验证/测试集，返回布尔掩码。
    确保每个类别中 train_ratio / val_ratio / test_ratio 的样本被分配到对应集合。

    参数：
        labels : torch.Tensor 或 numpy.ndarray，形状 (num_nodes,)，节点标签
        train_ratio, val_ratio, test_ratio : float，比例和应为 1
        seed : int，随机种子

    返回：
        train_mask, val_mask, test_mask : torch.BoolTensor，形状 (num_nodes,)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例和必须为1"
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    num_nodes = len(labels)
    rng = np.random.RandomState(seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 按类别分组索引
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    for class_id, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)  # 类内打乱
        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

    return train_mask, val_mask, test_mask

# 假设这些函数已存在（根据你的实际工具路径导入）
def load_XA(dataname, datadir="XAL"):
    prefix = os.path.join(datadir, dataname)
    filename_A = prefix + "_A.npy"
    filename_X = prefix + "_X.npy"
    A = np.load(filename_A)
    X = np.load(filename_X)
    return A, X


def load_labels(dataname, datadir="XAL"):
    prefix = os.path.join(datadir, dataname)
    filename_L = prefix + "_L.npy"
    L = np.load(filename_L)
    return L


def split_random_mask(num_nodes, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    随机划分训练/验证/测试集，返回布尔掩码。
    默认 80%/10%/10%，可调整。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    rng = np.random.RandomState(seed)
    indices = np.arange(num_nodes)
    rng.shuffle(indices)
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    return train_mask, val_mask, test_mask


def prepare_custom_dataset(dataset_name, dataset_path="dataset/", device=None):
    """
    加载自定义可解释GNN数据集，封装为PyG Data对象。
    参数：
        dataset_name: str, 如 "syn1", "syn2" 等
        dataset_path: 数据存放根目录
        device: torch.device, 若指定则将数据移至此设备
    返回：
        data: torch_geometric.data.Data
    """
    # 1. 加载邻接矩阵和特征（由原始数据接口提供）
    A, X = load_XA(dataset_name, datadir=dataset_path)  # 返回 np.array 或可转 np.array
    labels = load_labels(dataset_name, datadir=dataset_path)

    A = torch.tensor(A, dtype=torch.float32)
    X = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # 2. 根据数据集名称处理节点特征
    if dataset_name == "syn1":
        # syn1 包含额外特征，拼接度独热
        deg = torch.sum(A, dim=1).long()
        deg_onehot = F.one_hot(deg, num_classes=int(deg.max()) + 1).float()
        X = torch.cat([deg_onehot, X], dim=1)
    else:
        # 其他数据集仅使用度独热作为特征（根据原始处理逻辑）
        deg = torch.sum(A, dim=1).long()
        X = F.one_hot(deg, num_classes=int(deg.max()) + 1).float()

    # 3. 将稠密邻接矩阵转换为 COO 格式的边索引（无自环版本？这里保留原图）
    edge_index, _ = dense_to_sparse(A)
    edge_index = edge_index.long()

    # 4. 类别数
    num_classes = labels.max().item() + 1

    # 5. 随机划分（与 Cora 不同，此处无固定划分，每次运行重新随机）
    train_mask, val_mask, test_mask = split_random_mask(
        A.shape[0], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    # 6. 构建 PyG Data 对象
    data = Data(
        x=X,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # 可选：将数据移至指定设备
    if device is not None:
        data = data.to(device)

    # 额外保存类别数（可选）
    data.num_classes = num_classes

    return data


def prepare_data(dataset_name, dataset_path="dataset/"):
    A, X = load_XA(dataset_name, datadir=dataset_path)
    labels = load_labels(dataset_name, datadir=dataset_path)

    A = torch.tensor(A, dtype=torch.float32)
    X = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    if dataset_name == "syn1":
        one_hot = F.one_hot(torch.sum(A, 1).long()).float()
        X = torch.cat([one_hot, X], dim=1)
    else:
        X = F.one_hot(torch.sum(A, 1).long()).float()

    edge_index, _ = dense_to_sparse(A)

    num_classes = int(labels.max().item() + 1)
    labels_onehot = F.one_hot(labels, num_classes).float()

    return X, A, edge_index, labels, labels_onehot, num_classes


def main():
    cfg = Config('config.yaml')
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = Planetoid(root='./data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0].to(device)
#     data = prepare_custom_dataset("Syn1", dataset_path="./dataset/", device=device)
    cfg.in_feats = data.x.shape[1]
    data.train_mask, data.val_mask, data.test_mask = split_stratified_mask(data.y, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3)
    data = data.to(device)

    # ========== 半监督节点分类 ==========
    data_cls = data
    model_cls = EndToEndProtoModel_E(cfg, cfg.in_feats, data=data,num_classes=data.y.max().item() + 1).to(device)
    optimizer_cls = optim.Adam(model_cls.parameters(), lr=cfg.lr)


    best_acc = 0.0
    best_epoch = 0
    best_model_path = "best_model_cora.pt"

    losses_history = {
        'epochs': [],
        'align_loss': []
    }



    for epoch in range(500):
        if epoch<=10:
            loss = train_node_classification_distill_start(model_cls, data, optimizer_cls, cfg, full_batch=True, epoch=epoch)
            losses_history['align_loss'].append(loss['total'])
            losses_history['epochs'].append(epoch)
            print(f"Epoch: {epoch}, loss: {loss['total']}")
        else:
            loss = train_node_classification_distill(model_cls, data, optimizer_cls, cfg, full_batch=True, epoch=epoch)
            losses_history['align_loss'].append(loss['concept_align'])
            losses_history['epochs'].append(epoch)

        if (epoch + 1) % 10 == 0 and epoch > 10:
            # 使用示例（放在你训练循环的 test epoch 中）：
            print(f"Epoch: {epoch}")
            report, z, A = evaluate_model(model_cls, data, cfg, split='test')



if __name__ == "__main__":
    main()