import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_max
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# 添加自定义 scatter_softmax（兼容所有 PyG 版本）
def scatter_softmax(src, index, dim=0, eps=1e-12):
    """src: 任意形状张量，index: 同 dim 长度的一维索引"""
    max_val = scatter_max(src, index, dim=dim)[0][index]
    exp = torch.exp(src - max_val)
    sum_exp = scatter_sum(exp, index, dim=dim) + eps
    return exp / sum_exp[index]



class STAGATEAttnLayer(MessagePassing):
    """
    STAGATE 的图注意力层（无自环版本，如果原始图已包含自环也可使用）
    计算对称注意力系数：对于边 (i,j)，att_ij = softmax_j( sigmoid( f1_i + f2_j ) )
    其中 f1 = H @ v0, f2 = H @ v1
    """
    def __init__(self, in_dim):
        super(STAGATEAttnLayer, self).__init__(aggr='add')
        self.v0 = nn.Parameter(torch.FloatTensor(in_dim, 1))
        self.v1 = nn.Parameter(torch.FloatTensor(in_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.v0)
        nn.init.xavier_uniform_(self.v1)

    def forward(self, x, edge_index):
        # x: [N, in_dim]
        # 计算每个节点的两个标量
        f1 = x @ self.v0  # [N, 1]
        f2 = x @ self.v1  # [N, 1]

        # 在边上计算 logits
        # edge_index 形状 [2, E]，第0行是源节点，第1行是目标节点
        src, dst = edge_index
        logits = f1[src] + f2[dst]  # [E, 1]
        logits = torch.sigmoid(logits).squeeze(-1)  # [E]

        # 对每个目标节点进行 softmax 归一化（即稀疏 softmax）
        att = softmax(logits, dst)  # [E]

        # 使用注意力系数作为 edge_weight 进行消息传递
        # 注意：原始 STAGATE 传递的是节点特征 H，然后乘以注意力系数
        return self.propagate(edge_index, x=x, edge_weight=att), att

    def message(self, x_j, edge_weight):
        # x_j: [E, in_dim] 是邻居节点的特征
        # edge_weight: [E]
        return edge_weight.view(-1, 1) * x_j

class HyperNet(nn.Module):
    """小型超网络，从原型表示生成卷积核权重"""

    def __init__(self, proto_dim, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proto_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim * in_dim)
        )
        self.out_dim = out_dim
        self.in_dim = in_dim

    def forward(self, proto_vec):
        # proto_vec: (K, proto_dim)
        W = self.net(proto_vec)  # (K, out_dim * in_dim)
        W = W.view(-1, self.out_dim, self.in_dim)  # (K, out_dim, in_dim)
        return W


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax


class ProtoConditionedConv(nn.Module):
    """
    无 r 的原型条件图卷积。

    可解释逻辑：
    - 边 (u→v) 的变换矩阵 = Σ_k (A_uk · A_vk) · W_k
    - 即：两端节点对原型 k 的共认同程度，决定该原型卷积核在这条边上的权重
    - 邻居 u 的消息 = W_{u→v} · h_u
    """

    def __init__(self, in_dim, out_dim, K, proto_dim,
                 use_hypernet=True, hyper_hidden=64,
                 attn_mode='simple', num_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.proto_dim = proto_dim
        self.use_hypernet = use_hypernet
        self.attn_mode = attn_mode

        # 超网络：以概念表示为条件生成 K 个卷积核
        if use_hypernet:
            self.hyper = HyperNet(proto_dim * 2, out_dim, out_dim, hyper_hidden)
        else:
            self.W = nn.Parameter(torch.randn(K, out_dim, out_dim))

        # 区域适配因子 MLP
        self.region_adapt = nn.Sequential(
            nn.Linear( proto_dim + 1, 64),#in_dim + 2 * proto_dim + 1
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=-2)
        )

        # 残差连接
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
            self.self_proj = nn.Linear(out_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()
            self.self_proj = nn.Identity()

        # 注意力（可选）
        if attn_mode == 'gat':
            assert out_dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = out_dim // num_heads
            self.attn_a = nn.Parameter(torch.randn(num_heads, self.head_dim, 1))
        else:
            self.num_heads = 0

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h, edge_index, A_batch, P, C, proto_rep):
        N = h.size(0)
        row, col = edge_index
        K = P.shape[0]
        # d_proto = P.shape[1]
        # m_expand = h.unsqueeze(1).expand(N, K, h.shape[1])
        # P_expand = P.unsqueeze(0).expand(N, K, d_proto)
        # C_expand = C.unsqueeze(0).expand(N, K, d_proto)
        # A_append = A_batch.view(N,K,1)
        # mlp_input = torch.cat([A_append, P_expand], dim=-1)
        # r_vk = self.region_adapt(mlp_input/0.1).squeeze(-1)   # (N, K)





        # ---------- 1. 条件变换矩阵 ----------
        if self.use_hypernet:
            W_k = self.hyper(proto_rep)  # (K, out_dim, out_dim)
        else:
            W_k = self.W

        # 边条件化：两端原型共激活

        # A_row = A_batch[row]
        # A_col = A_batch[col]
        r_vk =  h @ P.t()
        A_row = r_vk[row]  # (E, K) target
        A_col = r_vk[col]  # (E, K) source
        edge_coeff = A_row * A_col  # (E, K)



        def compute_edge_weight_chunked(edge_coeff, W_k, chunk_size=4096):
            # edge_coeff: (E, k)
            E = edge_coeff.shape[0]
            outputs = []
            for i in range(0, E, chunk_size):
                # 每次只取 chunk_size 条边进入显存
                chunk_coeff = edge_coeff[i:i + chunk_size]  # (chunk, k)
                # 计算这个小块的 W_edge
                chunk_W = torch.einsum('ek,koi->eoi', chunk_coeff, W_k)  # (chunk, out_dim, out_dim)
                outputs.append(chunk_W)
                # 及时删除中间变量，保持显存平稳
                del chunk_W, chunk_coeff
                torch.cuda.empty_cache()  # 可选，视情况调用（太频繁会慢）
            return torch.cat(outputs, dim=0)
        # 可解释：如果 u 和 v 都强烈认同原型 k，则 W_k 主导这条边
        W_edge = torch.einsum('ek,koi->eoi', edge_coeff, W_k)  # (E, out_dim, out_dim)
        # W_edge = compute_edge_weight_chunked(edge_coeff, W_k)

        # ---------- 2. 消息生成 ----------
        h_src = h[col]  # (E, in_dim)
        if self.in_dim != self.out_dim:
            h_src = self.residual_proj(h_src)  # (E, out_dim)

        msg = torch.bmm(W_edge, h_src.unsqueeze(-1)).squeeze(-1)  # (E, out_dim)

        # ---------- 3. 聚合 ----------
        if self.attn_mode == 'simple':
            deg = scatter_sum(torch.ones_like(row, dtype=torch.float), row, dim_size=N) + 1e-10
            alpha_norm = 1.0 / deg[row]
            msg = alpha_norm.unsqueeze(-1) * msg

            m_neigh = scatter_sum(msg, row, dim=0, dim_size=N)
            self_msg = self.self_proj(self.residual_proj(h)) if self.in_dim != self.out_dim else h
            m_v = m_neigh + self_msg

        elif self.attn_mode == 'gat':
            msg_multi = msg.view(-1, self.num_heads, self.head_dim)
            attn_vec = self.attn_a.squeeze(-1)
            e = (msg_multi * attn_vec.unsqueeze(0)).sum(dim=-1)
            e = F.leaky_relu(e, 0.2)

            row_expand = row.unsqueeze(-1).repeat(1, self.num_heads).view(-1)
            e_flat = e.view(-1)
            alpha_flat = scatter_softmax(e_flat, row_expand, dim=0)
            alpha = alpha_flat.view(-1, self.num_heads, 1)

            msg_weighted = msg_multi * alpha
            out_flat = scatter_sum(
                msg_weighted.view(-1, self.head_dim),
                row_expand, dim=0, dim_size=N * self.num_heads
            )
            out_multi = out_flat.view(N, self.num_heads, self.head_dim)
            m_neigh = out_multi.reshape(N, self.out_dim)
            self_msg = self.self_proj(self.residual_proj(h)) if self.in_dim != self.out_dim else h
            m_v = m_neigh + self_msg
        else:
            raise ValueError(f"Unknown attn_mode: {self.attn_mode}")

        h_out = self.norm(m_v)

        r_vk = h_out @ P.t()
        return h_out,F.softmax(r_vk,dim=-1)


class AssignmentAdapter(nn.Module):
    """
    轻量残差分配更新器。
    A^{(l+1)} = softmax( (A^{(l)} + step_size · tanh(ΔA)) / τ )
    其中 ΔA 基于当前节点表示 h^(l+1) 计算。
    """

    def __init__(self, node_dim, K, tau=0.05, step_size=0.15):
        super().__init__()
        self.K = K
        self.tau = tau
        self.step_size = step_size

        self.delta_net = nn.Sequential(
            nn.Linear(node_dim + K, K),
            nn.LayerNorm(K),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(K, K)
        )
        # self.proto_persistence = nn.Parameter(torch.zeros(K))

    def forward(self, h, A_prev, tau = None):
        x = torch.cat([h, A_prev], dim=-1)  # (N, d+K)
        delta = self.delta_net(x)  # (N, K)

        # 残差更新：有界 + 先验偏置
        # A_perturbed = A_prev + self.step_size * torch.tanh(delta) + \
        #               self.proto_persistence.unsqueeze(0)
        A_perturbed = A_prev + self.step_size * torch.tanh(delta)

        if tau is None:
            A_new = F.softmax(A_perturbed / self.tau, dim=1)
        else:
            A_new = F.softmax(A_perturbed / tau, dim=1)
        return A_new

class LowRankHyperNet(nn.Module):
    def __init__(self, proto_dim, in_dim, out_dim, rank=8, hidden=64):
        super().__init__()
        self.rank = rank
        self.out_dim = out_dim
        self.in_dim = in_dim
        # 输出 U_k 和 V_k 的拼接：长度 = out_dim*rank + in_dim*rank
        self.net = nn.Sequential(
            nn.Linear(proto_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, (out_dim + in_dim) * rank)
        )

    def forward(self, proto_vec):
        # proto_vec: (K, proto_dim)
        params = self.net(proto_vec)                     # (K, (out+in)*r)
        U = params[:, :self.out_dim*self.rank].view(-1, self.out_dim, self.rank)   # (K, out_dim, r)
        V = params[:, self.out_dim*self.rank:].view(-1, self.in_dim, self.rank)    # (K, in_dim, r)
        return U, V