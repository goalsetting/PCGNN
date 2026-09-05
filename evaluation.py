import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from tqdm import tqdm


class ConceptExplainer:
    """
    将原型分配 A 转化为硬/软概念解释掩码。
    这相当于图级 SE-GNN 中的 explanation extractor e(G)=R。
    """

    def __init__(self, strategy='entropy_topk', k=3, tau=0.3):
        self.strategy = strategy
        self.k = k
        self.tau = tau

    def get_explanation_mask(self, A):
        """
        A: (N, K) 已归一化的分配矩阵
        返回: expl_mask (N, K)，1 表示该原型属于对应节点的概念解释
        """
        N, K = A.shape
        if self.strategy == 'topk':
            k_eff = min(self.k, K)
            _, idx = A.topk(k_eff, dim=-1)
            mask = torch.zeros_like(A)
            mask.scatter_(1, idx, 1.0)
        elif self.strategy == 'threshold':
            mask = (A > self.tau).float()
        elif self.strategy == 'entropy_topk':
            # 熵越低（置信越高），所需原型越少；这里简化仍用 topk，但可按熵加权
            k_eff = min(self.k, K)
            _, idx = A.topk(k_eff, dim=-1)
            mask = torch.zeros_like(A)
            mask.scatter_(1, idx, 1.0)
        else:
            raise ValueError
        return mask

    def get_complement_mask(self, A):
        return 1.0 - self.get_explanation_mask(A)


class ConceptFaithfulnessLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.explainer = ConceptExplainer(
            strategy=cfg.expl_strategy,
            k=cfg.expl_k,
            tau=cfg.expl_tau
        )

    def forward(self, model, data, z_full, A, h_smooth):
        """
        返回概念忠实度相关的各项损失。
        z_full: (N, C) 全原型前向 logits
        A: (N, K) 分配矩阵
        h_smooth: (N, d) 节点平滑表示（来自 forward_student）
        """
        device = z_full.device
        expl_mask = self.explainer.get_explanation_mask(A)  # (N, K)

        # ------------------- 1. 概念充分性损失 -------------------
        # 只用解释原型，预测应与全原型一致（对应 EST 的 R 本身）
        z_suff, _, _, _, _ = model.forward_student(
            data.x, data.edge_index, node_proto_mask=expl_mask,y= data.y, train_mask= data.train_mask
        )

        loss_suff = F.kl_div(
            z_suff[data.train_mask],
            z_full[data.train_mask].detach(),
            reduction='batchmean'
        )

        # ------------------- 2. 概念必要性损失 -------------------
        # 去掉解释原型后，预测应崩溃（验证解释不是冗余的）
        nec_mask = self.explainer.get_complement_mask(A)
        row_sum = nec_mask.sum(dim=1)
        valid = row_sum > 0  # 避免某节点解释包含全部原型

        if valid[data.train_mask].sum() > 0:
            z_nec, _, _, _, _ = model.forward_student(
                data.x, data.edge_index, node_proto_mask=nec_mask,y= data.y, train_mask= data.train_mask
            )
            # 最大化差异 = 最小化负 KL
            loss_nec = -F.kl_div(
                z_nec[data.train_mask & valid],
                z_full[data.train_mask & valid].detach(),
                reduction='batchmean'
            )
        else:
            loss_nec = torch.tensor(0.0, device=device)

        # ------------------- 3. 自适应掩码生成损失 -------------------
        # 让 adaptive_mask_gen 学会生成与 expl_mask 一致的掩码
        # 这样 mask 生成器本身就是可学习的"解释提取器"
        if hasattr(model, 'adaptive_mask_gen'):
            adaptive_mask = model.adaptive_mask_gen(h_smooth)  # (N, K)
            # 鼓励自适应掩码与硬解释一致（监督学习解释）
            loss_mask_align = F.binary_cross_entropy(
                adaptive_mask[data.train_mask],
                expl_mask[data.train_mask]
            )
            # 同时鼓励自适应掩码的充分性
            z_adapt, _, _, _, _ = model.forward_student(
                data.x, data.edge_index, node_proto_mask=adaptive_mask, A_full=A,y= data.y, train_mask= data.train_mask
            )
            loss_adapt_suff = F.kl_div(
                z_adapt[data.train_mask],
                z_full[data.train_mask].detach(),
                reduction='batchmean'
            )
        else:
            loss_mask_align = torch.tensor(0.0, device=device)
            loss_adapt_suff = torch.tensor(0.0, device=device)

        # ------------------- 4. 全局门控稀疏性 -------------------
        gate = torch.sigmoid(model.proto_gate)
        loss_sparse = gate.mean()  # L1 鼓励稀疏
        # 熵正则：鼓励 gate 趋近 0 或 1
        loss_gate_entropy = -(gate * torch.log(gate + 1e-10) +
                              (1 - gate) * torch.log(1 - gate + 1e-10)).mean()

        return {
            'loss_suff': loss_suff,
            'loss_nec': loss_nec,
            'loss_mask_align': loss_mask_align,
            'loss_adapt_suff': loss_adapt_suff,
            'loss_sparse': loss_sparse,
            'loss_gate_entropy': loss_gate_entropy,
            'expl_mask': expl_mask,  # 返回供可视化
            'adaptive_mask': adaptive_mask if hasattr(model, 'adaptive_mask_gen') else None
        }


class ConceptFaithfulnessEvaluator:
    """
    在概念/原型空间上计算忠实度指标。
    所有操作基于 A_student (N, K)，无需 edge_mask 或图子图枚举。
    """

    def __init__(self, model, cfg, num_est_samples: int = 30):
        self.model = model
        self.cfg = cfg
        self.num_est_samples = num_est_samples
        self.explainer = ConceptExplainer(
            strategy=cfg.expl_strategy,
            k=cfg.expl_k,
            tau=cfg.expl_tau
        )


    @torch.no_grad()
    def compute_edge_influence(self, data, mask=None, top_ratio=0.05):
        """
        模仿 compute_fid_minus_t 的指标计算过程，评估每个原型对应的边的重要性，
        最后对所有原型的结果取平均。

        参数:
            data: 图数据对象，包含 x, edge_index
            mask: 节点掩码 (可选)，默认为所有节点
            top_ratio: 裁剪比例，默认 0.2

        返回:
            (avg_high_diff, avg_low_diff):
                avg_high_diff : 删除高系数边后，预测不一致的平均比例
                avg_low_diff  : 删除低系数边后，预测不一致的平均比例
        """
        self.model.eval()
        _, _, _, _, A = self.model.forward_student(data.x, data.edge_index,y= data.y, train_mask= data.train_mask)
        # ------------- 第1步：计算 A (节点分配系数) -------------
        z_full, teacher_r_list, ths = self.model.forward_teacher(data.x, data.edge_index)



        # 完整图预测（用于计算差异）
        if mask is None:
            mask = torch.ones(data.x.size(0), dtype=torch.bool, device=data.x.device)
        pred_full = z_full[mask].argmax(dim=-1)

        N = data.x.size(0)
        K = A.size(1)
        num_nodes_to_select = max(1, int(N * top_ratio))  # 至少选1个节点

        # ---------- 2. 计算每个节点的熵 (分配均匀性) ----------
        # 防止log(0)，加极小值
        eps = 1e-10
        A_clamped = A.clamp(min=eps)
        entropy = -(A_clamped * torch.log(A_clamped)).sum(dim=1)  # (N,)

        # 获取全局熵最大的前 num_nodes_to_select 个节点索引 (低关联节点)
        _, low_node_indices = torch.topk(entropy, num_nodes_to_select, largest=True)
        # 直接转为张量，供后续向量化使用
        low_node_tensor = low_node_indices  # 已经是 tenso

        # 准备边信息
        edge_index = data.edge_index
        E = edge_index.size(1)

        # 累计差异
        total_high_diff = 0.0
        total_low_diff = 0.0

        # ---------- 3. 循环每个原型 ----------
        for k in tqdm(range(K), desc="Processing Prototypes", disable=(K < 10)):
            # 获取当前原型k的分配系数
            score_k = A[:, k]  # (N,)

            # 高关联节点：分配系数最大的前 num_nodes_to_select 个
            _, high_node_indices = torch.topk(score_k, num_nodes_to_select, largest=True)
            # 高关联节点张量 (当前原型)
            high_node_tensor = high_node_indices

            # ---------- 向量化删除边 ----------
            # 对于高关联节点：保留两端都不在 high_node_tensor 中的边
            is_high_connected = torch.isin(edge_index[0], high_node_tensor) | \
                                torch.isin(edge_index[1], high_node_tensor)
            keep_high = ~is_high_connected
            edge_high_removed = edge_index[:, keep_high]

            # 对于低关联节点（全局相同，直接使用预计算的 tensor）
            is_low_connected = torch.isin(edge_index[0], low_node_tensor) | \
                               torch.isin(edge_index[1], low_node_tensor)
            keep_low = ~is_low_connected
            edge_low_removed = edge_index[:, keep_low]

            # --- 在教师模型上重新前向传播 ---
            z_high_removed, _, _ = self.model.forward_teacher(data.x, edge_high_removed)
            z_low_removed, _, _ = self.model.forward_teacher(data.x, edge_low_removed)

            pred_high = z_high_removed[mask].argmax(dim=-1)
            pred_low = z_low_removed[mask].argmax(dim=-1)

            diff_high = (pred_full != pred_high).float().mean().item()
            diff_low = (pred_full != pred_low).float().mean().item()

            total_high_diff += diff_high
            total_low_diff += diff_low

        avg_high_diff = total_high_diff / K
        avg_low_diff = total_low_diff / K

        return avg_high_diff, avg_low_diff
    @torch.no_grad()
    def compute_fid_minus_t(self, data, mask: torch.Tensor) -> float:
        """
        Concept-Fid- (充分性):
        只用解释原型时，节点预测应保持不变。
        返回: sufficiency accuracy ∈ [0, 1]，越高越忠实。
        """
        self.model.eval()
        z_full, teacher_r_list, ths = self.model.forward_teacher(data.x, data.edge_index)

        hid = ths[-1]
        proj_h = hid
        # proj_h = self.model.teacher_proj[-1](hid)  # (N, out_dim)


        A = F.softmax(torch.mm(proj_h, self.model.prototypes.t()),dim=-1)

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 8))
        # plt.imshow(A.cpu().numpy(), cmap='viridis', aspect='auto', rasterized=True)
        # plt.colorbar(label='Value')
        # plt.show()

        expl_mask = self.explainer.get_explanation_mask(A)  # (N, K)

        if expl_mask is not None:
            if expl_mask.dim() == 1:
                expl_mask = expl_mask.unsqueeze(0).expand(A.size(0), -1)
            A = torch.mm(proj_h, self.model.prototypes.t()) * expl_mask
            # row_sum =A.sum(dim=1, keepdim=True) + 1e-10
            # A = A / row_sum

#@torch.inverse(self.model.teacher_proj[-1].weight)
        z_suff= F.softmax(self.model.teacher_encoder.node_classifier(torch.mm(A, self.model.prototypes)), dim=1)
        pred_full = z_full[mask].argmax(dim=-1)

        pred_suff = z_suff[mask].argmax(dim=-1)
        return (pred_full == pred_suff).float().mean().item()

    @torch.no_grad()
    def compute_fid_plus_t(self, data, mask: torch.Tensor) -> float:
        """
        Concept-Fid+ (必要性):
        去掉解释原型后，预测应发生显著变化。
        返回: necessity rate ∈ [0, 1]，越高越忠实。
        """
        self.model.eval()
        z_full, teacher_r_list, ths = self.model.forward_teacher(data.x, data.edge_index)

        hid = ths[-1]
        proj_h = hid
        # proj_h = self.model.teacher_proj[-1](hid)  # (N, out_dim)
        A = F.softmax(torch.mm(proj_h, self.model.prototypes.t()),dim=-1)
        expl_mask = self.explainer.get_explanation_mask(A)  # (N, K)
        comp_mask = 1.0 - expl_mask
        # 过滤掉解释包含全部原型的节点（无法计算必要性）
        row_sum = comp_mask.sum(dim=1)
        valid = row_sum > 0
        valid_mask = mask & valid
        if valid_mask.sum() == 0:
            return 0.0

        if comp_mask is not None:
            if comp_mask.dim() == 1:
                comp_mask = comp_mask.unsqueeze(0).expand(A.size(0), -1)
            A = torch.mm(proj_h, self.model.prototypes.t()) * comp_mask
            # row_sum =A.sum(dim=1, keepdim=True) + 1e-10
            # A = A / row_sum

        z_nec= F.softmax(self.model.teacher_encoder.node_classifier(torch.mm(A, self.model.prototypes)), dim=1)

        pred_full = z_full[valid_mask].argmax(dim=-1)
        pred_nec = z_nec[valid_mask].argmax(dim=-1)
        return (pred_full != pred_nec).float().mean().item()

    @torch.no_grad()
    def compute_fid_minus(self, data, mask: torch.Tensor) -> float:
        """
        Concept-Fid- (充分性):
        只用解释原型时，节点预测应保持不变。
        返回: sufficiency accuracy ∈ [0, 1]，越高越忠实。
        """
        self.model.eval()
        z_full, _, _, _, A = self.model.forward_student(data.x, data.edge_index,y= data.y, train_mask= data.train_mask)
        expl_mask = self.explainer.get_explanation_mask(A)  # (N, K)


        # edge_index = data.edge_index  # shape [2, E]
        # num_edges = edge_index.size(1)
        # num_to_remove = int(num_edges * 0.2)
        # perm = torch.randperm(num_edges)
        # keep_mask = torch.ones(num_edges, dtype=torch.bool)
        # keep_mask[perm[:num_to_remove]] = False
        # data.edgex_index = edge_index[:, keep_mask]
        # teacher_final, teacher_r_list, ths = self.model.forward_teacher(data.x, data.edgex_index)
        # self.model.teacher_x = ths[-1]
        z_suff, _, _, _, _ = self.model.forward_student(
            data.x, data.edge_index, node_proto_mask=expl_mask,y= data.y, train_mask= data.train_mask
        )

        pred_full = z_full[mask].argmax(dim=-1)
        pred_suff = z_suff[mask].argmax(dim=-1)
        return (pred_full == pred_suff).float().mean().item()

    @torch.no_grad()
    def compute_fid_plus(self, data, mask: torch.Tensor) -> float:
        """
        Concept-Fid+ (必要性):
        去掉解释原型后，预测应发生显著变化。
        返回: necessity rate ∈ [0, 1]，越高越忠实。
        """
        self.model.eval()
        z_full, _, _, _, A = self.model.forward_student(data.x, data.edge_index,y= data.y, train_mask= data.train_mask)
        expl_mask = self.explainer.get_explanation_mask(A)
        comp_mask = 1.0 - expl_mask

        # 过滤掉解释包含全部原型的节点（无法计算必要性）
        row_sum = comp_mask.sum(dim=1)
        valid = row_sum > 0
        valid_mask = mask & valid

        if valid_mask.sum() == 0:
            return 0.0

        z_nec, _, _, _, _ = self.model.forward_student(
            data.x, data.edge_index, node_proto_mask=comp_mask,y= data.y, train_mask= data.train_mask
        )

        pred_full = z_full[valid_mask].argmax(dim=-1)
        pred_nec = z_nec[valid_mask].argmax(dim=-1)
        return (pred_full != pred_nec).float().mean().item()

    @torch.no_grad()
    def compute_est(self, data, mask: torch.Tensor, distance: str = 'kl') -> float:
        """
        Concept-EST (扩展充分性测试):
        在概念空间采样超集：保留解释原型，随机加回部分补原型，
        测量最坏情况下的预测变化。

        distance: 'kl' | 'l1' | 'mse'
        返回: 平均最坏变化量，越低越忠实。
        """
        self.model.eval()
        z_full, _, _, _, A = self.model.forward_student(data.x, data.edge_index,y= data.y, train_mask= data.train_mask)
        expl_mask = self.explainer.get_explanation_mask(A)

        # 只考虑 A 中有非零权重的原型作为可扩展池
        comp_mask = (1.0 - expl_mask) * (A > 0.001).float()

        N = A.size(0)
        device = A.device
        max_diffs = torch.zeros(N, device=device)
        prob_full = F.softmax(z_full, dim=-1)
        pred_full = prob_full.argmax(dim=-1)


        for _ in range(self.num_est_samples):
            # 每个节点独立随机采样补原型的子集加入
            rand = torch.rand_like(comp_mask)
            # 随机比例 r ~ Uniform(0.2, 1.0)，决定加回多少补原型
            r = torch.rand(N, 1, device=device) * 0.8 + 0.2
            sampled_add = (rand < r).float() * comp_mask
            super_mask = ((expl_mask + sampled_add) > 0).float()

            z_super, _, _, _, _ = self.model.forward_student(
                data.x, data.edge_index, node_proto_mask=super_mask,y= data.y, train_mask= data.train_mask
            )
            prob_super = z_super

            if distance == 'kl':
#                 # KL(full || super)，衡量 super 偏离 full 的程度
#                 diff = F.kl_div(
#                     prob_super.log(), prob_full, reduction='none', log_target=False
#                 ).sum(dim=-1)
                pred_super = prob_super.argmax(dim=-1)
                diff =(pred_full != pred_super).float()
            elif distance == 'l1':
                diff = (prob_super - prob_full).abs().sum(dim=-1)
            else:  # mse
                diff = (prob_super - prob_full).pow(2).sum(dim=-1)

            max_diffs = torch.maximum(max_diffs, diff)

        return max_diffs[mask].mean().item()

    @torch.no_grad()
    def compute_gate_sparsity(self) -> Dict[str, float]:
        """评估全局门控的稀疏性与离散度"""
        gate = torch.sigmoid(self.model.proto_gate)
        return {
            'gate_sparsity': (gate < 0.1).float().mean().item(),
            'gate_active': (gate > 0.5).float().mean().item(),
            'gate_entropy': -(gate * torch.log(gate + 1e-10) +
                              (1 - gate) * torch.log(1 - gate + 1e-10)).mean().item()
        }

    @torch.no_grad()
    def evaluate(self, data, mask: torch.Tensor) -> Dict[str, float]:
        """一键返回所有概念忠实度指标"""
        A,B = self.compute_edge_influence(data, mask)
        metrics = {
            'teacher-Edge-Influnce': A / (A+B),
            'Concept-Fid-': self.compute_fid_minus(data, mask),
            'Concept-Fid+': self.compute_fid_plus(data, mask),
            'Concept-EST': self.compute_est(data, mask, distance='kl'),
            'Concept-EST-L1': self.compute_est(data, mask, distance='l1'),
        }
        metrics.update(self.compute_gate_sparsity())
        return metrics


class PrototypeInterpreter:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    @torch.no_grad()
    def get_prototype_semantics(self, data) -> Dict:
        """
        返回每个原型的：
        - gate 值（是否被全局启用）
        - 主导类别（通过 proto_classifier）
        - 训练集上的类别分布
        """
        self.model.eval()
        gate = torch.sigmoid(self.model.proto_gate).cpu().numpy()  # (K,)

        # 获取增强原型表示（使用当前数据分布的上下文）
        _, _, P_enh, C_enh, A = self.model.forward_student(data.x, data.edge_index,y= data.y, train_mask= data.test_mask)

        # 原型分类器倾向
        if hasattr(self.model, 'proto_classifier'):
            self.model.eval()
            logits = self.model.proto_classifier(P_enh)  # (K, num_classes)
            proto_class_scores = logits.cpu().numpy()
            proto_dominant_class = logits.argmax(dim=-1).cpu().numpy()
        else:
            proto_class_scores = None
            proto_dominant_class = None

        # 统计训练集上每个原型的"责任"：哪些类别的节点最依赖它
        train_A = A[data.test_mask].cpu().numpy()  # (N_train, K)
        train_y = data.y[data.test_mask].cpu().numpy()
        num_classes = int(data.y.max().item() + 1)
        K = A.size(1)

        # 对每个原型 k，计算它服务的各类别平均分配权重
        proto_class_responsibility = np.zeros((K, num_classes))
        for k in range(K):
            for c in range(num_classes):
                mask_c = (train_y == c)
                if mask_c.sum() > 0:
                    proto_class_responsibility[k, c] = train_A[mask_c, k].mean()
            # 归一化
            row_sum = proto_class_responsibility[k].sum()
            if row_sum > 0:
                proto_class_responsibility[k] /= row_sum

        return {
            'gate': gate,
            'proto_class_scores': proto_class_scores,
            'proto_dominant_class': proto_dominant_class,
            'proto_responsibility': proto_class_responsibility,
            'P_enh': P_enh.cpu().numpy(),
            'C_enh': C_enh.cpu().numpy(),
        }

    def visualize(self, data, save_path: str = None):
        """绘制原型语义仪表盘"""
        sem = self.get_prototype_semantics(data)
        K = len(sem['gate'])
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Proto Gate 值
        ax = axes[0, 0]
        colors = ['green' if g > 0.5 else 'gray' for g in sem['gate']]
        ax.bar(range(K), sem['gate'], color=colors)
        ax.axhline(0.5, color='red', linestyle='--', label='activation threshold')
        ax.set_xlabel('Prototype ID')
        ax.set_ylabel('Gate Value')
        ax.set_title('Global Prototype Gate (Learned Sparsity)')
        ax.set_ylim(0, 1)
        ax.legend()

        # 2. 原型-类别责任热力图
        ax = axes[0, 1]
        sns.heatmap(sem['proto_responsibility'], annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_xlabel('Class')
        ax.set_ylabel('Prototype')
        ax.set_title('Prototype-Class Responsibility (Train Set)')

        # 3. 原型分类器倾向（如果存在）
        if sem['proto_class_scores'] is not None:
            ax = axes[1, 0]
            sns.heatmap(sem['proto_class_scores'], annot=True, fmt='.1f', cmap='RdBu_r', ax=ax, center=0)
            ax.set_xlabel('Class')
            ax.set_ylabel('Prototype')
            ax.set_title('Proto-Classifier Logits')

        # 4. Gate vs 平均训练分配
        ax = axes[1, 1]
        avg_assignment = sem['proto_responsibility'].max(axis=1)  # 每个原型对某类的最大责任
        ax.scatter(avg_assignment, sem['gate'], c=range(K), cmap='tab10', s=100)
        for k in range(K):
            ax.annotate(str(k), (avg_assignment[k], sem['gate'][k]), fontsize=8)
        ax.set_xlabel('Max Class Responsibility')
        ax.set_ylabel('Gate Value')
        ax.set_title('Prototype Utility vs. Sparsity')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Saved] Prototype semantics plot -> {save_path}")
        plt.show()

    @torch.no_grad()
    def visualize_node_explanation(self, data, node_idx: int, save_path: str = None):
        """
        可视化单个节点的概念解释：
        - 该节点对各原型的分配权重
        - 全局门控
        - 高亮的解释原型（红色）
        """
        self.model.eval()
        _, _, _, _, A = self.model.forward_student(data.x, data.edge_index,y= data.y, train_mask= data.train_mask)
        explainer = ConceptExplainer(
            strategy=self.model.cfg.expl_strategy,
            k=self.model.cfg.expl_k,
            tau=self.model.cfg.expl_tau
        )
        expl_mask = explainer.get_explanation_mask(A)

        a_v = A[node_idx].cpu().numpy()
        gate = torch.sigmoid(self.model.proto_gate).cpu().numpy()
        expl_v = expl_mask[node_idx].cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(a_v))
        width = 0.35

        bars1 = ax.bar(x - width / 2, a_v, width, label='Node Assignment A[v]', alpha=0.8)
        bars2 = ax.bar(x + width / 2, gate, width, label='Global Gate', alpha=0.8)

        # 高亮被选为解释的原型
        for i in np.where(expl_v > 0)[0]:
            bars1[i].set_color('red')
            bars1[i].set_alpha(1.0)
            ax.annotate('EXPL', (i - width / 2, a_v[i]),
                        textcoords="offset points", xytext=(0, 5),
                        ha='center', fontsize=7, color='red')

        ax.set_xlabel('Prototype ID')
        ax.set_ylabel('Weight')
        ax.set_title(f'Node {node_idx} Concept Explanation (True y={data.y[node_idx].item()})')
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def evaluate_model(model, data, cfg, split: str = 'test') -> Tuple[Dict, torch.Tensor, torch.Tensor]:
    """
    完整评估入口。返回：
    - report: 指标字典
    - z: 最终预测 logits（供后续分析）
    - A: 原型分配矩阵（供可视化）
    """
    model.eval()
    mask = data.test_mask if split == 'test' else data.val_mask

    # ---------- 1. 基础性能 ----------
    with torch.no_grad():
        z, student_r_list, P_enh, C_enh, A = model.forward_student(data.x, data.edge_index,y= data.y, train_mask= data.train_mask)
        pred = z[mask].argmax(dim=-1)
        acc = (pred == data.y[mask]).float().mean().item()

    # ---------- 2. 概念忠实度 ----------
    faith_eval = ConceptFaithfulnessEvaluator(model, cfg, num_est_samples=30)
    faith_metrics = faith_eval.evaluate(data, mask)

    #     # ---------- 3. 教师-学生概念对齐 ----------
    #     with torch.no_grad():
    #         _, teacher_r_list = model.forward_teacher(data.x, data.edge_index)
    #     align_metrics = ConceptAlignmentEvaluator.compute_layerwise_alignment(
    #         student_r_list, teacher_r_list, mask
    #     )

    # ---------- 4. 原型语义 ----------
    interpreter = PrototypeInterpreter(model)
    proto_sem = interpreter.get_prototype_semantics(data)
    active_protos = int((proto_sem['gate'] > 0.5).sum())

    # ---------- 5. 组装报告 ----------
    report = {
        'split': split,
        'accuracy': acc,
        'active_prototypes': active_protos,
        'mean_gate': float(proto_sem['gate'].mean()),
    }
    report.update(faith_metrics)
    #     report.update(align_metrics)

    # 打印
    print(f"\n{'=' * 60}")
    print(f"Evaluation on {split.upper()} set")
    print(f"{'=' * 60}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k:30s}: {v:.4f}")
        else:
            print(f"{k:30s}: {v}")

    return report, z, A


# 使用示例（放在你训练循环的 test epoch 中）：
# report, z, A = evaluate_model(model, data, cfg, split='test')
#
# # 可视化原型语义
# interpreter = PrototypeInterpreter(model)
# interpreter.visualize(data, save_path='proto_semantics.png')
#
# # 诊断单个问题节点
# interpreter.visualize_node_explanation(data, node_idx=42, save_path='node_42_expl.png')


def get_ground_truth(node, dataset):
    gt = []
    if dataset == 'syn1':
        gt = get_ground_truth_syn1(node)  # correct
    elif dataset == 'syn2':
        gt = get_ground_truth_syn1(node)  # correct
    elif dataset == 'syn3':
        gt = get_ground_truth_syn3(node)  # correct
    elif dataset == 'syn4':
        gt = get_ground_truth_syn4(node)  # correct
    elif dataset == 'syn5':
        gt = get_ground_truth_syn5(node)  # correct
    elif dataset == 'syn6':
        gt = get_ground_truth_syn1(node)  # correct
    return gt


def get_ground_truth_syn1(node):
    base = [0, 1, 2, 3, 4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    return ground_truth


def get_ground_truth_syn3(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 3
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 3 for val in base]
    return ground_truth


def get_ground_truth_syn4(node):
    buff = node - 1
    base = [0, 1, 2, 3, 4, 5]
    ground_truth = []
    offset = buff % 6
    ground_truth = [buff - offset + val + 1 for val in base]
    return ground_truth


def get_ground_truth_syn5(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 7
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 7 for val in base]
    return ground_truth