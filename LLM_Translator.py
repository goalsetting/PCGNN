
"""
Cora 引文网络原型概念翻译管道
================================
基于 "统一骨架 + 场景血肉" 策略，实现从神经网络原型空间到人类语义空间的翻译。

依赖：
    pip install torch torch-geometric networkx numpy openai
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, to_networkx
import networkx as nx

from config import Config
from models import EndToEndProtoModel_E


# =============================================================================
# 一、Cora Meta 数据集管理
# =============================================================================

class CoraMetaManager:
    """
    Cora 数据集的 Meta 信息管理器。

    关键事实：
    ---------
    标准 PyG Planetoid 版本的 Cora 数据集使用 1433 维匿名二元词袋特征。
    原始数据文件不包含词汇表映射，特征维度为 w_0 ~ w_1432。

    解决方案：
    ---------
    1. 匿名模式（默认）：预计算每个特征索引与 7 个类别的关联强度，让 LLM 基于类别先验推断概念
    2. 增强模式（可选）：若用户提供词汇表文件（每行一个词，共1433行），自动加载并映射
    """

    CLASS_NAMES = [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
        'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'
    ]

    CLASS_NAMES_CN = [
        '基于案例的推理', '遗传算法', '神经网络',
        '概率方法', '强化学习', '规则学习', '理论'
    ]

    NUM_FEATURES = 1433
    NUM_NODES = 2708
    NUM_EDGES_UNDIRECTED = 5429

    def __init__(self, root: str = './data/Cora', vocab_file: Optional[str] = None):
        self.root = root
        self.vocab_file = vocab_file
        self._vocab: Optional[List[str]] = None
        self._data = None
        self._feature_class_profile: Optional[Dict[int, Dict]] = None

        # 加载词汇表（如果提供）
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self._vocab = [line.strip() for line in f.readlines()]
            assert len(self._vocab) == self.NUM_FEATURES, \
                f"词汇表大小必须为 {self.NUM_FEATURES}, 实际为 {len(self._vocab)}"

        # 预计算特征-类别关联档案
        self._build_feature_class_profile()

    @property
    def vocab(self) -> List[str]:
        """获取词汇表，匿名模式下返回特征索引名称"""
        if self._vocab is not None:
            return self._vocab
        return [f"feat_{i}" for i in range(self.NUM_FEATURES)]

    def load_data(self):
        """加载 PyG Planetoid Cora 数据集"""
        if self._data is None:
            dataset = Planetoid(root=self.root, name='Cora')
            self._data = dataset[0]
        return self._data

    def _build_feature_class_profile(self):
        """
        预计算每个特征维度与各类别的关联强度。
        用于匿名模式下推断"特征索引对应的语义方向"。
        """
        data = self.load_data()
        x, y = data.x, data.y

        profile = {}
        for feat_idx in range(self.NUM_FEATURES):
            feat_vector = x[:, feat_idx]
            class_activation = {}
            for c in range(len(self.CLASS_NAMES)):
                mask = (y == c)
                class_activation[c] = feat_vector[mask].float().mean().item() if mask.sum() > 0 else 0.0

            dominant_class = max(class_activation, key=class_activation.get)
            max_activation = class_activation[dominant_class]
            other_activations = [v for k, v in class_activation.items() if k != dominant_class]
            avg_other = sum(other_activations) / len(other_activations) if other_activations else 0
            discrimination = max_activation - avg_other

            profile[feat_idx] = {
                'dominant_class': dominant_class,
                'dominant_class_name': self.CLASS_NAMES[dominant_class],
                'activation_rate': max_activation,
                'discrimination': discrimination,
            }

        self._feature_class_profile = profile

    def get_feature_semantic_hint(self, feat_idx: int) -> str:
        """获取特征索引的语义提示（匿名模式下使用）"""
        if self._vocab is not None:
            return self._vocab[feat_idx]

        prof = self._feature_class_profile[feat_idx]
        return (f"feat_{feat_idx}"
                f"[关联:{prof['dominant_class_name']},"
                f"区分度:{prof['discrimination']:.3f}]")

    def get_class_name(self, class_idx: int, lang: str = 'en') -> str:
        """获取类别名称"""
        if lang == 'cn':
            return self.CLASS_NAMES_CN[class_idx]
        return self.CLASS_NAMES[class_idx]


# =============================================================================
# 二、显式统计特征与文本特征抽取
# =============================================================================

@dataclass
class PrototypeProfile:
    """原型统计画像数据结构"""
    proto_id: int
    gate_value: float
    top_nodes: List[int]
    top_scores: List[float]
    avg_degree: float
    degree_variance: float
    avg_clustering: float
    pagerank_mean: float
    betweenness_mean: float
    feature_signature: List[Tuple[str, float]]
    feature_class_bias: Dict[str, float]
    structural_signature: Dict[str, Any]
    dominant_pred_class: Optional[int]
    pred_class_distribution: Dict[int, float]
    neighbor_class_distribution: Dict[int, float]
    citation_hubs_ratio: float
    node_weights: Optional[torch.Tensor] = field(default=None, repr=False)  # 新增字段
    raw_top_nodes_tensor: Optional[torch.Tensor] = field(default=None, repr=False)


class CoraFeatureExtractor:
    """Cora 场景特征抽取器"""

    def __init__(self, meta_manager: CoraMetaManager):
        self.meta = meta_manager
        self._G_nx = None
        self._pagerank = None
        self._betweenness = None
        self._clustering = None

    def _ensure_graph_metrics(self, data):
        """预计算图结构指标（仅执行一次）"""
        if self._G_nx is None:
            self._G_nx = to_networkx(data, to_undirected=True)
            self._pagerank = nx.pagerank(self._G_nx)
            self._betweenness = nx.betweenness_centrality(self._G_nx)
            self._clustering = nx.clustering(self._G_nx)

    def extract_prototype_profile(
            self, model, data, proto_id: int,
            top_k_nodes: int = 20, top_k_features: int = 15
    ) -> PrototypeProfile:
        """
        抽取指定原型的完整统计画像。

        Args:
            model: 已训练的 ProtoGNN 模型（需有 forward_student 和 proto_gate）
            data: PyG Data 对象
            proto_id: 原型索引
            top_k_nodes: 高激活节点数
            top_k_features: 高激活特征数
        """
        self._ensure_graph_metrics(data)

        model.eval()
        with torch.no_grad():
            outputs = model.forward_student(data.x, data.edge_index, y= data.y, train_mask= data.train_mask)
            # 适配你的模型输出格式：(z, P, P_enh, C_enh, A, ...)
            A = outputs[4] if len(outputs) > 4 else outputs[-2]

            gate = torch.sigmoid(model.proto_gate)[proto_id].item() if hasattr(model, 'proto_gate') else 1.0

        # 1. 高激活节点（用于结构统计）
        scores, node_indices = A[:, proto_id].sort(descending=True)
        top_nodes = node_indices[:top_k_nodes]
        top_nodes_list = top_nodes.cpu().tolist()
        top_scores = scores[:top_k_nodes].cpu().tolist()

        # 2. 核心：计算节点维度的软权重（每个节点对原型的隶属度）
        proto_activation = torch.clamp(A[:, proto_id], min=0.0)
        node_weights = proto_activation / (proto_activation.sum() + 1e-12)

        # 3. 显式统计特征（结构） - 使用 top-k 节点
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        top_degrees = deg[top_nodes].cpu().numpy()
        avg_degree = float(top_degrees.mean())
        degree_variance = float(top_degrees.var())
        avg_clustering = float(np.mean([self._clustering.get(n, 0.0) for n in top_nodes_list]))
        pagerank_mean = float(np.mean([self._pagerank.get(n, 0.0) for n in top_nodes_list]))
        betweenness_mean = float(np.mean([self._betweenness.get(n, 0.0) for n in top_nodes_list]))

        # 4. 特征签名（使用节点软权重加权平均）
        feature_signature = self._extract_feature_signature(data, node_weights, top_k_features)
        feature_class_bias = self._analyze_feature_class_bias(feature_signature)

        # 5. 结构上下文
        structural_signature = self._extract_structural_signature(data, top_nodes, top_degrees)

        # 6. 预测关联（使用节点软权重加权类别分布）
        dominant_pred_class, pred_dist = self._extract_prediction_profile(data, node_weights)

        # 7. 邻居上下文（仍基于 top-k 节点）
        neighbor_dist, hubs_ratio = self._extract_neighborhood_summary(data, top_nodes)

        return PrototypeProfile(
            proto_id=proto_id, gate_value=gate, top_nodes=top_nodes_list,
            top_scores=top_scores, avg_degree=avg_degree,
            degree_variance=degree_variance, avg_clustering=avg_clustering,
            pagerank_mean=pagerank_mean, betweenness_mean=betweenness_mean,
            feature_signature=feature_signature, feature_class_bias=feature_class_bias,
            structural_signature=structural_signature,
            dominant_pred_class=dominant_pred_class, pred_class_distribution=pred_dist,
            neighbor_class_distribution=neighbor_dist, citation_hubs_ratio=hubs_ratio,
            node_weights=node_weights,  # 新增
            raw_top_nodes_tensor=top_nodes
        )

    def _extract_feature_signature(
            self, data, node_weights: torch.Tensor, top_k: int = 15
    ) -> List[Tuple[str, float]]:
        """
        使用节点维度的软权重对特征加权平均，提取词袋特征签名。
        """
        node_weights = node_weights.to(data.x.device)
        # data.x: [num_nodes, num_features]
        mean_feat = (data.x.float().T * node_weights).sum(dim=1).cpu().numpy()  # [num_features]
        top_indices = np.argsort(mean_feat)[::-1][:top_k]

        signature = []
        for idx in top_indices:
            activation = float(mean_feat[idx])
            if activation > 0:
                feat_name = self.meta.get_feature_semantic_hint(int(idx))
                signature.append((feat_name, activation))
        return signature

    def _analyze_feature_class_bias(
            self, feature_signature: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """分析特征签名的类别偏向性"""
        class_scores = {name: 0.0 for name in self.meta.CLASS_NAMES}

        for feat_hint, activation in feature_signature:
            match = re.search(r"关联:([^,\]]+)", feat_hint)
            if match:
                class_name = match.group(1).strip()
                if class_name in class_scores:
                    class_scores[class_name] += activation

        total = sum(class_scores.values())
        return {k: v / total for k, v in class_scores.items()} if total > 0 else class_scores

    def _extract_structural_signature(
            self, data, top_nodes: torch.Tensor, top_degrees: np.ndarray
    ) -> Dict[str, Any]:
        """提取学术影响力相关的结构签名"""
        top_nodes_list = top_nodes.cpu().tolist()
        q90 = np.quantile(top_degrees, 0.9)
        hubs_ratio = float((top_degrees > q90).mean())

        # 有向图分析：入度=被引，出度=引用
        out_deg = degree(data.edge_index[0], num_nodes=data.num_nodes)[top_nodes].cpu().numpy()
        in_deg = degree(data.edge_index[1], num_nodes=data.num_nodes)[top_nodes].cpu().numpy()

        avg_in = float(in_deg.mean())
        avg_out = float(out_deg.mean())

        return {
            'avg_citations_in': avg_in,
            'avg_citations_out': avg_out,
            'citation_ratio_in_out': avg_in / (avg_out + 1e-6),
            'hubs_ratio': hubs_ratio,
            'degree_heterogeneity': float(top_degrees.std() / (top_degrees.mean() + 1e-6)),
            'structural_role': self._infer_structural_role(
                avg_in, avg_out,
                float(np.mean([self._betweenness.get(n, 0) for n in top_nodes_list]))
            )
        }

    def _infer_structural_role(self, avg_in: float, avg_out: float, avg_bet: float) -> str:
        """推断该原型在引用网络中的结构角色"""
        if avg_in > 15 and avg_bet > 0.02:
            return "桥梁性高被引工作（可能是综述或跨领域基础论文）"
        elif avg_in > 10:
            return "领域内高影响力核心论文"
        elif avg_out > avg_in * 1.5:
            return "综述/引用密集型工作"
        elif avg_in < 5 and avg_out < 5:
            return "新兴/边缘研究方向"
        return "常规研究论文集群"

    def _extract_prediction_profile(
            self, data, node_weights: torch.Tensor
    ) -> Tuple[Optional[int], Dict[int, float]]:
        """
        使用节点维度的软权重计算原型类别隶属度。
        category_distribution[c] = sum_i w_i * 1{y_i == c}
        """
        if not hasattr(data, 'y') or data.y is None:
            return None, {}

        y = data.y
        num_classes = int(y.max().item()) + 1

        # 确保权重与标签在同一设备
        node_weights = node_weights.to(y.device)

        dist = {}
        for c in range(num_classes):
            mask = (y == c).float()
            dist[c] = float((node_weights * mask).sum().item())

        # 数值稳定性归一化
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}

        dominant = max(dist, key=dist.get) if dist else None
        return dominant, dist

    def _extract_neighborhood_summary(
            self, data, top_nodes: torch.Tensor
    ) -> Tuple[Dict[int, float], float]:
        """分析一阶邻居的类别分布"""
        if not hasattr(data, 'y') or data.y is None:
            return {}, 0.0

        top_nodes_set = set(top_nodes.cpu().tolist())
        neighbor_labels = []
        hub_neighbors = 0
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        q80 = torch.quantile(deg.float(), 0.8)

        for n in top_nodes.cpu().tolist():
            mask = data.edge_index[0] == n
            neighbors = data.edge_index[1][mask].cpu().tolist()
            for nb in neighbors:
                if nb not in top_nodes_set:
                    neighbor_labels.append(data.y[nb].item())
                    if deg[nb] > q80:
                        hub_neighbors += 1

        if not neighbor_labels:
            return {}, 0.0

        counter = Counter(neighbor_labels)
        total = len(neighbor_labels)
        dist = {k: v / total for k, v in counter.items()}
        hubs_ratio = hub_neighbors / total if total > 0 else 0.0
        return dist, float(hubs_ratio)


# =============================================================================
# 三、Prompt 模板：通用骨架 + Cora 血肉组装
# =============================================================================

class PromptTemplate:
    """Prompt 模板管理器"""

    UNIVERSAL_SKELETON = """你是一位 {domain_expert}。你的任务是将神经网络学习到的"原型"（prototype）翻译为可理解的人类概念。

## 角色定位
- 你不是在猜测，而是在基于统计证据进行科学归纳
- 你的命名必须体现 {domain_specific_constraint}
- 禁止编造数据中没有的信息

## 输入格式
该原型的统计画像如下：

### 基础信息
- 原型编号：{proto_id}
- 模型重要性（gate值）：{gate_value:.4f}（范围0-1，越高越重要）
- 高激活节点数：{num_top_nodes}

### 特征签名（关键词/特征模式）
{feature_signature}

### 特征类别偏向性
{feature_class_bias}

### 结构上下文
{structural_signature}

### 预测关联（节点标签分布）
{prediction_profile}

### 邻居上下文（一阶邻域标签分布）
{neighborhood_summary}

## 任务要求
请基于以上统计画像，完成以下输出：

1. **命名**：给出一个 2-6 字的中文概念名 + 英文翻译，要求专业、准确、无歧义。
2. **定义**：用 1-2 句话描述这个概念的本质（提炼因果机制，不是罗列特征）。
3. **行为画像**：这个原型代表的节点"长什么样"？（给领域人员一个快速识别模板）
4. **决策指南**：当模型输出某节点属于此原型时，领域人员应该关注什么？采取什么行动？
5. **置信度**：你对这个命名的确信程度（高/中/低），并说明理由。

## 约束条件
- 禁止编造数据中没有的信息
- 命名必须体现{domain_specific_constraint}
- 如果画像信息不足，明确说明"需要更多上下文"，不要强行命名
- 输出必须是严格的 JSON 格式，不要包含任何 Markdown 代码块标记之外的内容

## 输出格式
{output_schema}
"""

    CORA_FEATURE_SIGNATURE_FMT = (
        "Cora 节点特征是 1433 维二元词袋（Bag-of-Words）。\n"
        "以下特征按平均激活强度排序（强度越高，该词在该原型中越具代表性）：\n"
        "{items}\n"
        "注：若使用匿名特征，'feat_N[关联:ClassName,区分度:D]' 表示该特征在 ClassName 类论文中高激活，"
        "区分度 D 越高说明该特征越专属于该类别。"
    )

    CORA_STRUCTURAL_FMT = (
        "在学术引用网络中，结构指标对应以下语义：\n"
        "- 平均被引数（入度）：{avg_citations_in:.2f} → 学术影响力\n"
        "- 平均引用数（出度）：{avg_citations_out:.2f} → 知识吸收广度\n"
        "- 被引/引用比：{citation_ratio_in_out:.2f} → >1 为被关注型，<1 为综述型\n"
        "- 高被引论文比例：{hubs_ratio:.2%} → 核心工作占比\n"
        "- 度异质性：{degree_heterogeneity:.2f} → 越高说明该原型内部影响力差异越大\n"
        "- 局部聚类系数：{avg_clustering:.4f} → 越高说明研究圈子越紧密\n"
        "- PageRank 均值：{pagerank_mean:.6f} → 全局影响力\n"
        "- Betweenness 均值：{betweenness_mean:.6f} → 桥梁性（跨领域连接能力）\n"
        "- 结构角色推断：{structural_role}\n"
        "- 邻居中高被引节点比例：{neighbor_hubs_ratio:.2%} → 与核心工作的邻近度"
    )

    CORA_PREDICTION_FMT = (
        "该原型高激活节点的 ground-truth 类别分布：\n"
        "{class_dist}\n"
        "主导类别：{dominant_class}（占比 {dominant_ratio:.1%}）\n"
        "注：Cora 7 个类别为：Case_Based(基于案例推理), Genetic_Algorithms(遗传算法), "
        "Neural_Networks(神经网络), Probabilistic_Methods(概率方法), "
        "Reinforcement_Learning(强化学习), Rule_Learning(规则学习), Theory(理论)"
    )

    CORA_NEIGHBOR_FMT = (
        "一阶邻居（引用/被引）的类别分布：\n"
        "{neighbor_dist}\n"
        "邻居类别熵：{entropy:.3f}（越高说明该原型越处于跨领域交界）\n"
        "与核心工作相邻比例：{hubs_ratio:.2%}"
    )

    CORA_EXAMPLES = """
## 命名示例参考
- 特征偏向 Neural_Networks + 高被引 + 高 Betweenness → "深度学习基础理论" / "Deep Learning Foundations"
- 特征偏向 Genetic_Algorithms + 紧密聚类 + 中等被引 → "进化计算核心方法" / "Evolutionary Computation Core"
- 特征混合 Neural_Networks + Probabilistic_Methods + 高邻居熵 → "神经概率建模" / "Neural Probabilistic Modeling"
- 特征偏向 Theory + 低被引 + 高引用出度 → "形式化方法综述" / "Formal Methods Survey"
- 高被引 + 跨类别邻居 + 高 PageRank → "桥梁性综述工作" / "Bridge Survey Work"

## 命名禁忌
❌ 禁止使用纯统计描述：如"高激活节点群"、"feat_42 主导群"
❌ 禁止生造词：如"Neural-Genetic 混合体"
❌ 禁止过度推断：如"这是图灵奖级别的工作"（数据无法支撑）

## 特别注意
1. 区分"方法论创新"（提出新算法/新理论框架）和"应用研究"（将现有方法用于新领域）
2. 如果关键词混合了理论和应用，优先考虑"应用驱动的方法论适配"
3. 如果结构签名显示高被引且跨领域连接多，优先考虑"桥梁性/综述性"定位
4. 如果局部聚类系数很高但 PageRank 中等，可能是"紧密的细分子领域"
"""

    OUTPUT_SCHEMA = """
{
  "concept_name": "中文概念名（2-6字）",
  "concept_name_en": "English Concept Name",
  "definition": "用1-2句话描述本质因果机制",
  "node_profile": "该原型节点的识别模板（给审稿人/编辑的快速判断依据）",
  "actionable_guide": "当模型判定某论文属于此原型时，领域人员应采取的行动/关注点",
  "confidence": "高|中|低",
  "confidence_reason": "置信度理由，必须引用具体数据支撑"
}
"""

    @classmethod
    def build_cora_prompt(cls, profile: PrototypeProfile, meta: CoraMetaManager) -> str:
        # 特征签名
        feat_items = "\n".join([
            f"  - {name}: 激活强度={act:.4f}"
            for name, act in profile.feature_signature
        ])
        feature_signature_text = cls.CORA_FEATURE_SIGNATURE_FMT.format(items=feat_items)

        # 类别偏向性
        bias_items = "\n".join([
            f"  - {cls_name}: {score:.3f}"
            for cls_name, score in profile.feature_class_bias.items()
        ])
        feature_bias_text = f"该原型的特征在以下类别中偏向性分布：\n{bias_items}"

        # 结构签名
        ss = profile.structural_signature
        structural_text = cls.CORA_STRUCTURAL_FMT.format(
            avg_citations_in=ss['avg_citations_in'],
            avg_citations_out=ss['avg_citations_out'],
            citation_ratio_in_out=ss['citation_ratio_in_out'],
            hubs_ratio=ss['hubs_ratio'],
            degree_heterogeneity=ss['degree_heterogeneity'],
            avg_clustering=profile.avg_clustering,
            pagerank_mean=profile.pagerank_mean,
            betweenness_mean=profile.betweenness_mean,
            structural_role=ss['structural_role'],
            neighbor_hubs_ratio=profile.citation_hubs_ratio
        )

        # 预测关联
        if profile.dominant_pred_class is not None:
            dominant_name = meta.get_class_name(profile.dominant_pred_class)
            dominant_ratio = profile.pred_class_distribution.get(profile.dominant_pred_class, 0)
            class_dist = "\n".join([
                f"  - {meta.get_class_name(k)}: {v:.1%}"
                for k, v in sorted(profile.pred_class_distribution.items())
            ])
            prediction_text = cls.CORA_PREDICTION_FMT.format(
                class_dist=class_dist, dominant_class=dominant_name, dominant_ratio=dominant_ratio
            )
        else:
            prediction_text = "无标签信息"

        # 邻居上下文
        if profile.neighbor_class_distribution:
            neighbor_dist = "\n".join([
                f"  - {meta.get_class_name(k)}: {v:.1%}"
                for k, v in sorted(profile.neighbor_class_distribution.items())
            ])
            probs = list(profile.neighbor_class_distribution.values())
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            neighborhood_text = cls.CORA_NEIGHBOR_FMT.format(
                neighbor_dist=neighbor_dist, entropy=entropy,
                hubs_ratio=profile.citation_hubs_ratio
            )
        else:
            neighborhood_text = "无邻居标签信息"

        # 组装
        prompt = cls.UNIVERSAL_SKELETON.format(
            domain_expert="计算科学文献计量学家与机器学习领域专家",
            domain_specific_constraint="学术领域的内在逻辑（如'方法论创新'与'应用研究'的区分，'理论奠基'与'实证分析'的层次差异），不能是 superficial 的关键词堆砌",
            proto_id=profile.proto_id,
            gate_value=profile.gate_value,
            num_top_nodes=len(profile.top_nodes),
            feature_signature=feature_signature_text,
            feature_class_bias=feature_bias_text,
            structural_signature=structural_text,
            prediction_profile=prediction_text,
            neighborhood_summary=neighborhood_text,
            output_schema=cls.OUTPUT_SCHEMA
        )
        prompt += f"\n\n{cls.CORA_EXAMPLES}"
        return prompt


# =============================================================================
# 四、LLM 客户端（支持 OpenAI / vLLM / Ollama）
# =============================================================================

class LLMClient:
    def __init__(
            self, backend: str = "openai", api_key: Optional[str] = None,
            base_url: Optional[str] = None, model: str = "gpt-4o",
            temperature: float = 0.3, max_retries: int = 3
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        if backend == "openai":
            import openai
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        elif backend == "vllm":
            import openai
            self.client = openai.OpenAI(api_key="EMPTY", base_url=base_url or "http://localhost:8000/v1")
        elif backend == "ollama":
            self.client = None
            self.base_url = base_url or "http://localhost:11434"
        else:
            raise ValueError(f"不支持的 backend: {backend}")

    def generate(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                if self.backend in ("openai", "vllm"):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=self.temperature,
                        max_tokens=2048
                    )
                    return response.choices[0].message.content
                elif self.backend == "ollama":
                    import requests
                    resp = requests.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model, "prompt": prompt,
                            "format": "json", "stream": False,
                            "options": {"temperature": self.temperature}
                        }
                    )
                    return resp.json()["response"]
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM 调用失败: {e}")
                print(f"[Retry {attempt + 1}] {e}")
        return "{}"


# =============================================================================
# 五、端到端原型概念翻译器
# =============================================================================

class CoraPrototypeConceptualizer:
    """
    使用流程：
    1. meta = CoraMetaManager(root='./data/Cora')
    2. llm = LLMClient(backend="openai", api_key="...", model="gpt-4o")
    3. translator = CoraPrototypeConceptualizer(model, data, llm, meta)
    4. guide = translator.generate_field_guide('cora_field_guide.json')
    """

    def __init__(
            self, model, data, llm_client: LLMClient,
            meta_manager: Optional[CoraMetaManager] = None,
            gate_threshold: float = 0.1, top_k_nodes: int = 200,
            top_k_features: int = 15
    ):
        self.model = model
        self.data = data
        self.llm = llm_client
        self.meta = meta_manager or CoraMetaManager()
        self.extractor = CoraFeatureExtractor(self.meta)
        self.gate_threshold = gate_threshold
        self.top_k_nodes = top_k_nodes
        self.top_k_features = top_k_features

    def get_all_prototype_profiles(self) -> List[PrototypeProfile]:
        """提取所有激活原型的统计画像"""
        # 推断原型数量 K
        if hasattr(self.model, 'K'):
            K = self.model.K
        elif hasattr(self.model, 'num_prototypes'):
            K = self.model.num_prototypes
        else:
            with torch.no_grad():
                outputs = self.model.forward_student(self.data.x, self.data.edge_index,y= self.data.y, train_mask= self.data.train_mask)
                A = outputs[4] if len(outputs) > 4 else outputs[-2]
                K = A.size(1)

        profiles = []
        for k in range(K):
            if hasattr(self.model, 'proto_gate'):
                gate = torch.sigmoid(self.model.proto_gate)[k].item()
                if gate < self.gate_threshold:
                    continue

            try:
                profile = self.extractor.extract_prototype_profile(
                    self.model, self.data, k,
                    self.top_k_nodes, self.top_k_features
                )
                profiles.append(profile)
            except Exception as e:
                print(f"[Warning] 原型 {k} 抽取失败: {e}")
                continue

        profiles.sort(key=lambda p: p.gate_value, reverse=True)
        return profiles

    def conceptualize(self, profile: PrototypeProfile) -> Dict:
        """调用 LLM 将原型翻译为人类概念"""
        prompt = PromptTemplate.build_cora_prompt(profile, self.meta)
        # raw_response = self.llm.generate(prompt)
        raw_response = prompt
        cleaned = prompt
        # # 清理 Markdown 代码块
        # cleaned = re.sub(r'^```json\s*', '', raw_response.strip())
        # cleaned = re.sub(r'```\s*$', '', cleaned)

        try:
            concept = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            concept = json.loads(match.group()) if match else {
                "concept_name": "解析失败", "concept_name_en": "Parse Error",
                "definition": f"LLM 输出无法解析。原始输出：{raw_response[:200]}...",
                "node_profile": "N/A", "actionable_guide": "请检查 LLM 输出格式",
                "confidence": "低", "confidence_reason": "JSON 解析失败"
            }

        concept['_meta'] = {
            'proto_id': profile.proto_id,
            'gate_value': profile.gate_value,
            'dominant_class': profile.dominant_pred_class,
            'dominant_class_name': (
                self.meta.get_class_name(profile.dominant_pred_class)
                if profile.dominant_pred_class is not None else None
            )
        }
        return concept

    def generate_field_guide(self, output_path: str) -> Dict:
        """生成完整的领域工作人员指南（Field Guide）"""
        profiles = self.get_all_prototype_profiles()
        print(f"[Info] 共发现 {len(profiles)} 个激活原型")

        concepts = []
        for profile in profiles:
            print(f"[Info] 正在翻译原型 {profile.proto_id} (gate={profile.gate_value:.4f})...")
            concepts.append(self.conceptualize(profile))

        guide = {
            'scenario': 'cora_academic_citation',
            'dataset': 'Cora',
            'model_info': {
                'num_active_prototypes': len(profiles),
                'gate_threshold': self.gate_threshold
            },
            'concepts': concepts,
            'usage_guide': self._build_usage_guide(concepts)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(guide, f, indent=2, ensure_ascii=False)
        print(f"[Field Guide Generated] {output_path}")
        return guide

    def _build_usage_guide(self, concepts: List[Dict]) -> str:
        """构建面向领域人员的使用指南"""
        lines = [
            "=== Cora 学术引用网络原型概念使用指南 ===\n",
            "【决策流程】",
            "1. 对目标论文，查看其 A_student 向量中权重 Top-2 的原型",
            "2. 若 Top-1 原型置信度为'高'且 gate > 0.8，可直接采用其概念标签",
            "3. 若 Top-2 原型权重接近（差 < 0.1），说明该论文处于交叉领域",
            "4. 若所有原型置信度均为'低'，说明可能是新兴方向，建议人工审查\n",
            "【审稿/编辑场景建议】",
            "- 方法论创新类原型 → 匹配方法论专家审稿",
            "- 应用研究类原型 → 匹配应用领域专家审稿",
            "- 桥梁性综述类原型 → 匹配跨领域资深编辑",
        ]
        return "\n".join(lines)


# =============================================================================
# 六、快速启动示例
# =============================================================================

if __name__ == "__main__":
    cfg = Config('config.yaml')
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # 1. 加载 Meta 和数据
    meta = CoraMetaManager(root='./data/Cora')
    data = meta.load_data()
    data = data.to(device)

    # 2. 加载你的原型模型（示例，请替换为你的实际模型）
    cfg.in_feats = data.x.shape[1]
    model = EndToEndProtoModel_E(cfg, cfg.in_feats, data=data,num_classes=data.y.max().item() + 1).to(device)
    # num_classes = data.y.max().item() + 1
    # model.classifier = torch.nn.Linear(cfg.pc_out_dim, num_classes).to(device)
    model.load_state_dict(torch.load('best_model_cora.pt'))
    model.eval()

    # # 3. 初始化 LLM（示例：OpenAI）
    # llm = LLMClient(
    #     backend="openai",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model="gpt-4o",
    #     temperature=0.3
    # )
    llm = None

    # 4. 运行翻译器
    translator = CoraPrototypeConceptualizer(
        model=model, data=data, llm_client=llm,
        meta_manager=meta, gate_threshold=0.1
    )
    guide = translator.generate_field_guide('cora_field_guide.json')

    print("请根据你的实际模型接口调整 model.forward_student 的调用位置")
