import torch
from torch import nn


class FeatureProjection(nn.Module):
    def __init__(self, in_dim, out_dim, data):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # 用 PCA 初始化线性层权重
        self._init_pca_weights(data.x)

    def _init_pca_weights(self, x):
        # 计算 PCA 的投影矩阵（前 out_dim 个主成分方向）
        x_centered = x - x.mean(dim=0)
        U, S, V = torch.pca_lowrank(x_centered, q=self.linear.out_features)
        self.linear.weight.data = V[:, :self.linear.out_features].t()  # 注意形状对齐
        self.linear.bias.data = - (x.mean(dim=0) @ self.linear.weight.t())

    def forward(self, x):
        return self.linear(x)

def kmeans(x, ncluster, niter=50):
    '''
    x : torch.tensor(data_num, data_dim)
    ncluster : 聚类数量
    niter : 迭代次数
    Returns:
        c : (ncluster, data_dim) 聚类中心
        labels : (data_num,) 每个样本的簇标签 (0 ~ ncluster-1)
    '''
    N, D = x.size()
    # 随机初始化聚类中心
    c = x[torch.randperm(N)[:ncluster]]

    for i in range(niter):
        # 计算每个样本到各中心的距离，并分配最近簇
        a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
        # 更新聚类中心（按簇取均值）
        c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
        # 处理空簇（NaN 中心）：用随机样本重新初始化
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        if ndead > 0:
            c[nanix] = x[torch.randperm(N)[:ndead]]

    # 最后再计算一次标签，确保与最终中心对应（处理空簇重新初始化后的情况）
    labels = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)

    return c, labels