# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # 以 GraphSAGE 为例
from torch_scatter import scatter_mean

from base_model import GNNEncoder, ContextMLP, GraphDiffusion
from layers import ProtoConditionedConv, AssignmentAdapter, HyperNet, STAGATEAttnLayer
from pretrain import load_or_pretrain_semi_gnn
from pretrain_gene import load_or_pretrain_stagate
from sinkhorn import sinkhorn_ot



# ============================================================
# 4. 主模型（完全重写，维度与调用逻辑全修正）
# ============================================================
class EndToEndProtoModel_E_Gene(nn.Module):
    def __init__(self, cfg, num_node_features, num_classes=0, data=None):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.num_prototypes
        self.num_classes = num_classes

        # ---------- 1. 用于计算 A 的投影层（必须与 prototypes 同维）----------
        # 关键：prototypes 是 (K, out_dim)，所以 feat_proj 必须输出 out_dim
        self.feat_proj = nn.Linear(num_node_features, cfg.out_dim)

        # 可选扩散：仅用于增强结构感知以计算更稳定的 A
        if getattr(cfg, 'use_diffusion', True):
            self.diffusion = GraphDiffusion(cfg.diffusion_alpha)
        else:
            self.diffusion = nn.Identity()

        # ---------- 2. 原型与上下文 ----------
        self.prototypes = nn.Parameter(torch.randn(self.K, cfg.out_dim))
        self.context_mlp = ContextMLP(cfg.out_dim, cfg.context_hidden)
        self.semantic_emb = nn.Parameter(torch.randn(self.K, cfg.out_dim), requires_grad=False)

        self.teacher_x = None

        # ---------- 3. 概念图 ----------
        self.concept_graph = ProtoConceptGraph(cfg, self.prototypes, None, None)

        # ---------- 4. 条件 GNN（关键修正）----------
        # in_dim 是第一层卷积维度（pc_hidden_dim），不是原始特征维度！
        # ProtoConditionedGNN 内部通过 proj_x 自行处理原始 x
        self.conditioned_gnn = ProtoConditionedGNN(
            cfg,
            in_dim=cfg.out_dim,       # 第一层卷积维度
            K=self.K,
            proto_dim=cfg.out_dim,            # P_enh 的维度
            data=data
        )

        # ---------- 5. 全局门控 ---------- nn.Parameter(torch.zeros(self.K)) torch.zeros(self.K).to(torch.device('cuda'))
        self.proto_gate = nn.Parameter(torch.zeros(self.K))

        # ---------- 6. 教师模型（改为自监督 STAGATE，冻结）----------
        self.teacher_encoder = load_or_pretrain_stagate(
            data, data.x.device, num_classes = num_classes, save_path="stagate_self_supervised.pt",
            hidden_dims=getattr(cfg, 'teacher_hidden_dims', [cfg.out_dim, cfg.out_dim]),  # 可配置，默认与原 STAGATE 一致
            lr=getattr(cfg, 'teacher_lr', 0.001),
            n_epochs=getattr(cfg, 'teacher_epochs', 1000),
            weight_decay=getattr(cfg, 'teacher_weight_decay', 0.0001),
            gradient_clipping=getattr(cfg, 'teacher_grad_clip', 5.0),
            nonlinear=getattr(cfg, 'teacher_nonlinear', True),
            verbose=True
        )
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        # 教师投影头：根据 STAGATE 编码器各层输出维度动态创建
        # STAGATE 的 W 是 ParameterList，每个元素形状 [in_dim, out_dim]
        teacher_layer_out_dims = [w.shape[1] for w in self.teacher_encoder.W]
        self.teacher_proj = nn.ModuleList([
            nn.Linear(dim, cfg.out_dim) for dim in teacher_layer_out_dims
        ])

        # ---------- 7. 节点分类头 ----------
        self.node_classifier = nn.Linear(cfg.out_dim, num_classes)

        # ---------- 8. 辅助 ----------
        self.prototype_readout = nn.Linear(cfg.out_dim * 2, cfg.out_dim)
        self.tau_cert = cfg.tau_cert

    # ------------------------------------------------------------------
    # tau 调度（与学生 AssignmentAdapter、教师共享同一套参数）
    # ------------------------------------------------------------------
    def get_tau_schedule(self, num_layers):
        base = getattr(self.cfg, 'assign_tau_base', 0.10)
        decay = getattr(self.cfg, 'assign_tau_decay', 0.60)
        return [base * (decay ** l) for l in range(num_layers)]

    # ------------------------------------------------------------------
    # 学生前向（核心逻辑修正）
    # ------------------------------------------------------------------
    def forward_student(self, x, edge_index, A_full=None, node_proto_mask=None,
                        return_h_smooth=False, use_gate=True, y=None, train_mask=None):
        """
        严格流程：
          1. x → feat_proj → h_for_A (out_dim)
          2. h_for_A → diffusion → h_diffused（仅用于计算 A）
          3. h_diffused 与 prototypes 做余弦相似度 → Sinkhorn → A_init
          4. conditioned_gnn 接收原始 x（由内部 proj_x 自行投影到 pc_hidden_dim）
          5. conditioned_gnn 内部用 A_init 驱动所有条件消息传递
        """
        # Step 1: 投影到原型空间（用于余弦相似度）
        h_for_A = self.feat_proj(x)                         # (N, out_dim)

        # Step 2: 可选扩散（仅用于获得结构感知的 A）
        h_diffused = self.diffusion(h_for_A, edge_index)    # (N, out_dim)

        # Step 3: 计算初始分配 A_init
        if A_full is None:
            if self.teacher_x is not None:
                h_norm = F.normalize(self.teacher_x, p=2, dim=1)    # (N, out_dim)
            else:
                h_norm = F.normalize(h_diffused, p=2, dim=1)    # (N, out_dim)

            proto_norm = F.normalize(self.prototypes, p=2, dim=1)  # (K, out_dim)
            cos_sim = torch.mm(h_norm, proto_norm.t())     # (N, K), [-1, 1]
            cost = 1.0 - cos_sim                            # [0, 2], 越小越相似

            # 构建先验 R
            R = torch.ones(cos_sim.shape[0], cos_sim.shape[1], device=cos_sim.device)   # 默认均匀
            if train_mask is not None and train_mask.any():
                # 计算原型-类别分布 q
                A_train = cost[train_mask]  # 当前分配的软矩阵（需要历史或重新计算）
                y_train = y[train_mask]
                q = torch.zeros(cos_sim.shape[1], y.max().item() + 1, device=cos_sim.device)
                for c in range(y.max().item() + 1):
                    mask_c = (y_train == c)
                    if mask_c.any():
                        q[:, c] = A_train[mask_c].sum(dim=0)
                q = q / (q.sum(dim=1, keepdim=True) + 1e-10)

                # 对每个有标签节点，设置先验
                for v in train_mask.nonzero(as_tuple=True)[0]:
                    c = y[v]
                    R[v] = q[:, c]  # 直接使用 q_{k,c}
                # R[train_mask] = R[train_mask] / R[train_mask].sum(dim=1, keepdim=True)  # 行归一化

            A_init = sinkhorn_ot(
                cost, self.cfg.ot_epsilon,
                max_iter=self.cfg.ot_max_iter,
                llm_prior=R,
                tol=self.cfg.ot_tol
            )
        else:
            A_init = A_full

        # Step 4: 门控与掩码
        if use_gate:
            gate = torch.sigmoid(self.proto_gate)           # (K,)
            A_init = A_init * gate.unsqueeze(0)

        if node_proto_mask is not None:
            if node_proto_mask.dim() == 1:
                node_proto_mask = node_proto_mask.unsqueeze(0).expand(A_init.size(0), -1)
            A_init = A_init * node_proto_mask

        # 行归一化（节点视角，用于 conditioned_gnn 的边条件化）
        row_sum = A_init.sum(dim=1, keepdim=True) + 1e-10
        A_row = A_init / row_sum

        # 列归一化（原型视角，用于上下文聚合）
        col_sum = A_init.sum(dim=0, keepdim=True) + 1e-10
        A_col = A_init / col_sum

        # Step 5: 上下文与概念图
        c = torch.mm(A_col.t(), self.context_mlp(h_diffused))
        P_enh, C_enh, stack_PC = self.concept_graph_forward(
            self.prototypes, c, self.semantic_emb, A_row
        )

        if use_gate:
            P_enh = P_enh * gate.unsqueeze(-1)
            C_enh = C_enh * gate.unsqueeze(-1)
            stack_PC = stack_PC * gate.unsqueeze(-1)

        # Step 6: 条件 GNN（关键修正：传入原始 x，不是 h_diffused！）
        # conditioned_gnn 内部通过 proj_x 自行投影到 pc_hidden_dim
        h_out, A_list, hs = self.conditioned_gnn(
            x, edge_index, A_row, P_enh, C_enh, stack_PC
        )

        # Step 7: 最终预测
        z = F.softmax(self.node_classifier(h_out), dim=1)

        if return_h_smooth:
            return z, A_list, P_enh, C_enh, A_row, hs
        return z, A_list, P_enh, C_enh, A_row

    # ------------------------------------------------------------------
    # 教师前向（适配自监督 STAGATE）
    # ------------------------------------------------------------------
    def forward_teacher(self, x, edge_index, num_align_layers=None):
        self.teacher_encoder.eval()
        hiddens = []
        H = x
        # 手动逐层执行 STAGATE 编码器
        for layer in range(len(self.teacher_encoder.W)):
            H = H @ self.teacher_encoder.W[layer]
            if layer < len(self.teacher_encoder.attn_layers):
                # 有注意力层
                H, _ = self.teacher_encoder.attn_layers[layer](H, edge_index)
                if self.teacher_encoder.nonlinear:
                    H = F.elu(H)
            # 最后一层无注意力和激活，直接添加
            hiddens.append(H)

        # 对齐层数
        student_num_layers = len(self.conditioned_gnn.layers)
        num_align_layers = num_align_layers or min(
            len(hiddens), len(self.teacher_proj), student_num_layers
        )

        teacher_A_list = []
        with torch.no_grad():
            for l in range(num_align_layers):
                hid = hiddens[l]
                proj_h = self.teacher_proj[l](hid)          # (N, out_dim)
                sim = torch.mm(proj_h, self.prototypes.t()) / self.cfg.tau
                A_teacher = F.sigmoid(sim)      # (N, K)
                teacher_A_list.append(A_teacher)

        # 最终教师表示：使用最后一层投影后的特征（不再经过分类器）
        teacher_final = self.teacher_proj[-1](hiddens[-1])  # (N, out_dim)
        # 注意：teacher_final 不再 softmax，而是原型空间特征，用于反向对齐损失
        return teacher_final, teacher_A_list, hiddens

    # ------------------------------------------------------------------
    # 概念对齐损失（对齐 A_list）
    # ------------------------------------------------------------------
    def compute_concept_align_loss(self, student_A_list, teacher_A_list):
        min_len = min(len(student_A_list), len(teacher_A_list))
        if min_len == 0:
            return torch.tensor(0.0, device=student_A_list[0].device)

        total_kl = 0
        for sA, tA in zip(student_A_list[:min_len], teacher_A_list[:min_len]):
            kl = F.kl_div(
                (sA + 1e-10).log(),
                tA.detach(),
                reduction='batchmean'
            )
            total_kl += kl

        return total_kl / min_len

    # ------------------------------------------------------------------
    # 反向回灌（保留）
    # ------------------------------------------------------------------
    def compute_reverse_alignment_sample(self, teacher_final, P_enh, C_enh):
        # teacher_final 现在为 (N, out_dim) 特征向量
        teacher_proto_sim = torch.mm(teacher_final, self.prototypes.t()) / self.cfg.tau
        teacher_A_final = F.softmax(teacher_proto_sim, dim=1)
        global_proto = self.prototype_readout(torch.cat([P_enh, C_enh], dim=-1))
        teacher_global = torch.mm(teacher_A_final, global_proto)
        return F.mse_loss(teacher_global, teacher_final)

    # ------------------------------------------------------------------
    # 概念图前向（保留原有逻辑）
    # ------------------------------------------------------------------
    def concept_graph_forward(self, P, C, E, A_full):
        if self.cfg.knn_k == 0:
            P_out = P
            C_out = C
            x_out = torch.cat([P, C], dim=-1)
        else:
            K = P.size(0)
            sim_list = []

            if self.cfg.use_struct_sim and A_full is not None:
                col_norm = A_full.pow(2).sum(0, keepdim=True) + 1e-10
                cooc = torch.mm(A_full.t(), A_full)
                struct_sim = cooc / torch.sqrt(torch.mm(col_norm.t(), col_norm))
                sim_list.append(struct_sim)

            if self.cfg.use_context_sim:
                C_norm = F.normalize(C, p=2, dim=1)
                ctx_sim = torch.mm(C_norm, C_norm.t())
                sim_list.append(ctx_sim)

            if self.cfg.use_semantic_sim and E is not None:
                E_norm = F.normalize(E, p=2, dim=1)
                sem_sim = torch.mm(E_norm, E_norm.t())
                sim_list.append(sem_sim)

            if len(sim_list) == 0:
                sim = torch.eye(K, device=P.device)
            else:
                if self.cfg.cg_fusion_mode == 'attention':
                    weights = torch.softmax(self.concept_graph.attn_fusion, dim=0)
                    sim = sum(w * s for w, s in zip(weights, sim_list))
                else:
                    sim = torch.stack(sim_list, dim=0).sum(0)

            topk_idx = torch.topk(sim, self.cfg.knn_k, dim=-1, largest=True)[1]
            mask = torch.zeros_like(sim)
            mask.scatter_(1, topk_idx, 1.0)
            sim = sim * mask
            sim = (sim + sim.t()) / 2

            if self.cfg.cg_adj_mode == 'sym':
                deg = sim.sum(dim=1, keepdim=True) + 1e-10
                adj = deg.pow(-0.5) * sim * deg.pow(-0.5)
                mask = (adj == 0)
            else:
                attn_mask = (sim == 0).to(sim.dtype) * -1e9
                adj = F.softmax(sim + attn_mask, dim=-1)
                mask = (adj == 0)

            x = torch.cat([P, C], dim=-1)
            x_out = self.concept_graph.transformer(x.unsqueeze(0), mask=mask)
            x_out = x_out.squeeze(0)
            d = self.cfg.out_dim
            P_out = x_out[:, :d]
            C_out = x_out[:, d:] if x_out.size(1) > d else C

        return P_out, C_out, x_out




# ============================================================
# 4. 主模型（完全重写，维度与调用逻辑全修正）
# ============================================================
class EndToEndProtoModel_E(nn.Module):
    def __init__(self, cfg, num_node_features, num_classes=0, data=None):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.num_prototypes
        self.num_classes = num_classes

        # ---------- 1. 用于计算 A 的投影层（必须与 prototypes 同维）----------
        # 关键：prototypes 是 (K, out_dim)，所以 feat_proj 必须输出 out_dim
        self.feat_proj = nn.Linear(num_node_features, cfg.out_dim)

        # 可选扩散：仅用于增强结构感知以计算更稳定的 A
        if getattr(cfg, 'use_diffusion', True):
            self.diffusion = GraphDiffusion(cfg.diffusion_alpha)
        else:
            self.diffusion = nn.Identity()

        # ---------- 2. 原型与上下文 ----------
        self.prototypes = nn.Parameter(torch.randn(self.K, cfg.out_dim))
        self.context_mlp = ContextMLP(cfg.out_dim, cfg.context_hidden)
        self.semantic_emb = nn.Parameter(torch.randn(self.K, cfg.out_dim), requires_grad=False)

        self.teacher_x = None

        # ---------- 3. 概念图 ----------
        self.concept_graph = ProtoConceptGraph(cfg, self.prototypes, None, None)

        # ---------- 4. 条件 GNN（关键修正）----------
        # in_dim 是第一层卷积维度（pc_hidden_dim），不是原始特征维度！
        # ProtoConditionedGNN 内部通过 proj_x 自行处理原始 x
        self.conditioned_gnn = ProtoConditionedGNN(
            cfg,
            in_dim=cfg.out_dim,       # 第一层卷积维度
            K=self.K,
            proto_dim=cfg.out_dim,            # P_enh 的维度
            data=data
        )

        # ---------- 5. 全局门控 ---------- nn.Parameter(torch.zeros(self.K)) torch.zeros(self.K).to(torch.device('cuda'))
        self.proto_gate = nn.Parameter(torch.zeros(self.K))

        # ---------- 6. 教师模型（冻结）----------
        self.teacher_encoder = load_or_pretrain_semi_gnn(
            cfg, data, data.x.device, save_path="semi_GNN.pt"
        )
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        # 教师投影头：映射到学生原型空间（cfg.out_dim）
        self.teacher_proj = nn.ModuleList([
            nn.Linear(cfg.hidden_dim if i < cfg.num_layers - 1 else cfg.out_dim, cfg.out_dim)
            for i in range(cfg.num_layers)
        ])

        # ---------- 7. 节点分类头 ----------
        self.node_classifier = nn.Linear(cfg.out_dim, num_classes)

        # ---------- 8. 辅助 ----------
        self.prototype_readout = nn.Linear(cfg.out_dim * 2, cfg.out_dim)
        self.tau_cert = cfg.tau_cert

    # ------------------------------------------------------------------
    # tau 调度（与学生 AssignmentAdapter、教师共享同一套参数）
    # ------------------------------------------------------------------
    def get_tau_schedule(self, num_layers):
        base = getattr(self.cfg, 'assign_tau_base', 0.10)
        decay = getattr(self.cfg, 'assign_tau_decay', 0.60)
        return [base * (decay ** l) for l in range(num_layers)]

    # ------------------------------------------------------------------
    # 学生前向（核心逻辑修正）
    # ------------------------------------------------------------------
    def forward_student(self, x, edge_index, A_full=None,node_proto_mask=None,
                        return_h_smooth=False, use_gate=True, y = None, train_mask = None):
        """
        严格流程：
          1. x → feat_proj → h_for_A (out_dim)
          2. h_for_A → diffusion → h_diffused（仅用于计算 A）
          3. h_diffused 与 prototypes 做余弦相似度 → Sinkhorn → A_init
          4. conditioned_gnn 接收原始 x（由内部 proj_x 自行投影到 pc_hidden_dim）
          5. conditioned_gnn 内部用 A_init 驱动所有条件消息传递
        """
        # Step 1: 投影到原型空间（用于余弦相似度）
        h_for_A = self.feat_proj(x)                         # (N, out_dim)

        # Step 2: 可选扩散（仅用于获得结构感知的 A）
        h_diffused = self.diffusion(h_for_A, edge_index)    # (N, out_dim)

        # Step 3: 计算初始分配 A_init
        if A_full is None:
            # if self.teacher_x is not None:
            #     h_norm = F.normalize(self.teacher_x, p=2, dim=1)    # (N, out_dim)
            # else:
            #     h_norm = F.normalize(h_diffused, p=2, dim=1)    # (N, out_dim)
            h_norm = F.normalize(h_diffused, p=2, dim=1)  # (N, out_dim)
            proto_norm = F.normalize(self.prototypes, p=2, dim=1)  # (K, out_dim)
            cos_sim = torch.mm(h_norm, proto_norm.t())     # (N, K), [-1, 1]
            cost = 1.0 - cos_sim                            # [0, 2], 越小越相似

            # 构建先验 R
            R = torch.ones(cos_sim.shape[0], cos_sim.shape[1], device=cos_sim.device)   # 默认均匀
            if train_mask.any():
                # 计算原型-类别分布 q
                A_train = cost[train_mask]  # 当前分配的软矩阵（需要历史或重新计算）
                y_train = y[train_mask]
                q = torch.zeros(cos_sim.shape[1], y.max().item() + 1, device=cos_sim.device)
                for c in range(y.max().item() + 1):
                    mask_c = (y_train == c)
                    if mask_c.any():
                        q[:, c] = A_train[mask_c].sum(dim=0)
                q = q / (q.sum(dim=1, keepdim=True) + 1e-10)

                # 对每个有标签节点，设置先验
                for v in train_mask.nonzero(as_tuple=True)[0]:
                    c = y[v]
                    R[v] = q[:, c]  # 直接使用 q_{k,c}
                # R[train_mask] = R[train_mask] / R[train_mask].sum(dim=1, keepdim=True)  # 行归一化

            A_init = sinkhorn_ot(
                cost, self.cfg.ot_epsilon,
                max_iter=self.cfg.ot_max_iter,
                llm_prior = R,
                tol=self.cfg.ot_tol
            )
        else:
            A_init = A_full

        # Step 4: 门控与掩码
        if use_gate:
            gate = torch.sigmoid(self.proto_gate)           # (K,)
            A_init = A_init * gate.unsqueeze(0)

        if node_proto_mask is not None:
            if node_proto_mask.dim() == 1:
                node_proto_mask = node_proto_mask.unsqueeze(0).expand(A_init.size(0), -1)
            A_init = A_init * node_proto_mask

        # 行归一化（节点视角，用于 conditioned_gnn 的边条件化）
        row_sum = A_init.sum(dim=1, keepdim=True) + 1e-10
        A_row = A_init / row_sum

        # 列归一化（原型视角，用于上下文聚合）
        col_sum = A_init.sum(dim=0, keepdim=True) + 1e-10
        A_col = A_init / col_sum

        # Step 5: 上下文与概念图
        c = torch.mm(A_col.t(), self.context_mlp(h_diffused))
        P_enh, C_enh, stack_PC = self.concept_graph_forward(
            self.prototypes, c, self.semantic_emb, A_row
        )

        if use_gate:
            P_enh = P_enh * gate.unsqueeze(-1)
            C_enh = C_enh * gate.unsqueeze(-1)
            stack_PC = stack_PC * gate.unsqueeze(-1)

        # Step 6: 条件 GNN（关键修正：传入原始 x，不是 h_diffused！）
        # conditioned_gnn 内部通过 proj_x 自行投影到 pc_hidden_dim
        h_out, A_list, hs = self.conditioned_gnn(
            x, edge_index, A_row, P_enh, C_enh, stack_PC
        )


        # Step 7: 最终预测
        z = F.softmax(self.node_classifier(h_out),dim=1)

        if return_h_smooth:
            return z, A_list, P_enh, C_enh, A_row, hs
        return z, A_list, P_enh, C_enh, A_row

    # ------------------------------------------------------------------
    # 教师前向（tau 调度与学生共享）
    # ------------------------------------------------------------------
    def forward_teacher(self, x, edge_index, num_align_layers=None):
        self.teacher_encoder.eval()
        hiddens = []
        h = x
        for i, conv in enumerate(self.teacher_encoder.convs):
            h = conv(h, edge_index)
            if i < self.teacher_encoder.num_layers - 1:
                h = F.relu(h)
                if hasattr(self.teacher_encoder, 'bns') and i < len(self.teacher_encoder.bns):
                    h = self.teacher_encoder.bns[i](h)
                h = self.teacher_encoder.dropout(h)
            hiddens.append(h)

        # 对齐层数
        student_num_layers = len(self.conditioned_gnn.layers)
        num_align_layers = num_align_layers or min(
            len(hiddens), len(self.teacher_proj), student_num_layers
        )
        #
        # # 共享 tau 调度
        # tau_list = self.get_tau_schedule(num_align_layers)
        #
        teacher_A_list = []
        with torch.no_grad():
            for l in range(num_align_layers):
                hid = hiddens[l]
                proj_h = hid
                # proj_h = self.teacher_proj[l](hid)          # (N, out_dim)
                sim = torch.mm(proj_h, self.prototypes.t())
                A_teacher = F.softmax(sim,dim=-1)
                # A_teacher = F.sigmoid(sim)      # (N, K)
                teacher_A_list.append(A_teacher)

        teacher_final = F.softmax(self.teacher_encoder.node_classifier(hiddens[-1]), dim=1)
        return teacher_final, teacher_A_list, hiddens

    # ------------------------------------------------------------------
    # 概念对齐损失（对齐 A_list）
    # ------------------------------------------------------------------
    def compute_concept_align_loss(self, student_A_list, teacher_A_list):
        min_len = min(len(student_A_list), len(teacher_A_list))
        if min_len == 0:
            return torch.tensor(0.0, device=student_A_list[0].device)

        total_kl = 0
        for sA, tA in zip(student_A_list[:min_len], teacher_A_list[:min_len]):
            kl = F.kl_div(
                (sA + 1e-10).log(),
                tA.detach(),
                reduction='batchmean'
            )
            total_kl += kl

        return total_kl / min_len

    # ------------------------------------------------------------------
    # 反向回灌（保留）
    # ------------------------------------------------------------------
    def compute_reverse_alignment_sample(self, teacher_final, P_enh, C_enh):
        teacher_proto_sim = torch.mm(teacher_final, self.prototypes.t()) / self.cfg.tau
        teacher_A_final = F.softmax(teacher_proto_sim, dim=1)
        global_proto = self.prototype_readout(torch.cat([P_enh, C_enh], dim=-1))
        teacher_global = torch.mm(teacher_A_final, global_proto)
        return F.mse_loss(teacher_global, teacher_final)

    # ------------------------------------------------------------------
    # 概念图前向（保留原有逻辑）
    # ------------------------------------------------------------------
    def concept_graph_forward(self, P, C, E, A_full):
        if self.cfg.knn_k == 0:
            P_out = P
            C_out = C
            x_out = torch.cat([P, C], dim=-1)
        else:
            K = P.size(0)
            sim_list = []

            if self.cfg.use_struct_sim and A_full is not None:
                col_norm = A_full.pow(2).sum(0, keepdim=True) + 1e-10
                cooc = torch.mm(A_full.t(), A_full)
                struct_sim = cooc / torch.sqrt(torch.mm(col_norm.t(), col_norm))
                sim_list.append(struct_sim)

            if self.cfg.use_context_sim:
                C_norm = F.normalize(C, p=2, dim=1)
                ctx_sim = torch.mm(C_norm, C_norm.t())
                sim_list.append(ctx_sim)

            if self.cfg.use_semantic_sim and E is not None:
                E_norm = F.normalize(E, p=2, dim=1)
                sem_sim = torch.mm(E_norm, E_norm.t())
                sim_list.append(sem_sim)

            if len(sim_list) == 0:
                sim = torch.eye(K, device=P.device)
            else:
                if self.cfg.cg_fusion_mode == 'attention':
                    weights = torch.softmax(self.concept_graph.attn_fusion, dim=0)
                    sim = sum(w * s for w, s in zip(weights, sim_list))
                else:
                    sim = torch.stack(sim_list, dim=0).sum(0)

            topk_idx = torch.topk(sim, self.cfg.knn_k, dim=-1, largest=True)[1]
            mask = torch.zeros_like(sim)
            mask.scatter_(1, topk_idx, 1.0)
            sim = sim * mask
            sim = (sim + sim.t()) / 2

            if self.cfg.cg_adj_mode == 'sym':
                deg = sim.sum(dim=1, keepdim=True) + 1e-10
                adj = deg.pow(-0.5) * sim * deg.pow(-0.5)
                mask = (adj == 0)
            else:
                attn_mask = (sim == 0).to(sim.dtype) * -1e9
                adj = F.softmax(sim + attn_mask, dim=-1)
                mask = (adj == 0)

            x = torch.cat([P, C], dim=-1)
            x_out = self.concept_graph.transformer(x.unsqueeze(0), mask=mask)
            x_out = x_out.squeeze(0)
            d = self.cfg.out_dim
            P_out = x_out[:, :d]
            C_out = x_out[:, d:] if x_out.size(1) > d else C

        return P_out, C_out, x_out

class ProtoConceptGraph(nn.Module):
    """
    构建原型概念图并进行图推理。
    输入：质心 P, 上下文向量 C, 语义嵌入 E (K, d), 软映射表 A (N, K)
    输出：增强后的原型表示 P_out, C_out
    """

    def __init__(self, cfg, P, C, E):
        super().__init__()

        self.K = P.size(0)
        self.d = P.size(1)

        if C is not None:
            # 注册为 buffer，不参与梯度更新，因为我们假设原型已冻结
            self.register_buffer('P', P)
            self.register_buffer('C', C)

        self.knn_k = cfg.knn_k
        self.use_struct = cfg.use_struct_sim
        self.use_context = cfg.use_context_sim
        self.use_semantic = cfg.use_semantic_sim
        self.fusion = cfg.cg_fusion_mode

        if E is not None:
            self.register_buffer('E', E)
        else:
            self.E = None

        # 融合权重（若为 attention 模式则可学习）
        if self.fusion == 'attention':
            self.attn_fusion = nn.Parameter(torch.ones(2) / 2)

        # 概念图推理网络：使用标准 Transformer Encoder
        # 输入特征为 P 和 C 的拼接，或仅用其中之一
        self.proj_in = nn.Linear(self.d * 2, self.d)  # 拼接投影
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d * 2,
                nhead=2,
                dim_feedforward=self.d * 2,
                dropout=cfg.cg_dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=cfg.cg_num_layers
        )
        # 输出投影回原维度
        self.proj_P = nn.Linear(self.d, self.d)
        self.proj_C = nn.Linear(self.d, self.d)

    def build_adj(self, A):

        """利用软映射 A 和原型静态特征构建概念图邻接矩阵 (K, K) 稀疏化"""
        sim_list = []
        # 结构共现相似度
        if self.use_struct and A is not None:
            # A: (N, K)
            col_norm = A.pow(2).sum(0, keepdim=True) + 1e-10  # (1, K)
            cooc = torch.mm(A.t(), A)  # (K, K)
            struct_sim = cooc / torch.sqrt(torch.mm(col_norm.t(), col_norm))  # 余弦
            sim_list.append(struct_sim)

        # 区域上下文相似度
        if self.use_context:
            C_norm = F.normalize(self.C, p=2, dim=1)
            ctx_sim = torch.mm(C_norm, C_norm.t())
            sim_list.append(ctx_sim)

        # 语义相似度
        if self.use_semantic and hasattr(self, 'E') and self.E is not None:
            E_norm = F.normalize(self.E, p=2, dim=1)
            sem_sim = torch.mm(E_norm, E_norm.t())
            sim_list.append(sem_sim)

        if len(sim_list) == 0:
            return torch.eye(self.K, device=self.P.device)

        # 融合多模态相似度
        if self.fusion == 'attention':
            weights = torch.softmax(self.attn_fusion, dim=0)
            sim = sum(w * s for w, s in zip(weights, sim_list))
        else:  # add 或 max
            sim = torch.stack(sim_list, dim=0)
            if self.fusion == 'add':
                sim = sim.sum(0)
            elif self.fusion == 'max':
                sim = sim.max(0)[0]

        # k-NN 稀疏化 + 对称归一化
        topk_vals, topk_idx = torch.topk(sim, self.knn_k, dim=-1, largest=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk_idx, 1.0)
        sim = sim * mask
        # 对称化
        sim = (sim + sim.t()) / 2
        # 行归一化（有向边权重）
        deg_inv_sqrt = torch.pow(sim.sum(dim=1, keepdim=True) + 1e-10, -0.5)
        adj_norm = deg_inv_sqrt * sim * deg_inv_sqrt
        return adj_norm

    def forward(self, A_full=None):
        """
        A_full: 可选，全量软映射表 (N, K)，用于构建概念图。
                若为 None，则仅使用静态原型特征构建。
        """

        if self.knn_k == 0:
            # Transformer 的掩码在 batch_first=True 时形状 (K, K) 需扩展 batch
            P_out = self.P
            C_out = self.C
            x_out = self.P
        else:
            # 拼接特征 [P || C]
            x = torch.cat([self.P, self.C], dim=-1)  # (K, 2d)
            adj = self.build_adj(A_full)  # (K, K)
            # Transformer 推理，需要 mask 来体现图结构（可选）
            # 这里使用邻接矩阵的边作为注意力掩码 (True/False) 或不使用
            # 为简洁，我们直接让 Transformer 全连接，或通过 mask 限制
            # 可传入 src_mask 但 Transformer 期望掩码形状 (S, S)
            # 我们制作一个布尔掩码：非邻居设为 -inf 注意力
            mask = (adj == 0).to(x.device)  # (K, K) True 表示不连接
            # Transformer 的掩码在 batch_first=True 时形状 (K, K) 需扩展 batch
            x_out = self.transformer(x.unsqueeze(0), mask=mask)  # (1, K, 2d)
            x_out = x_out.squeeze(0)  # (K, 2d)
            P_out = x_out[:, :self.d]
            C_out = self.C

        return P_out, C_out, x_out


class ProtoConditionedGNN(nn.Module):
    def __init__(self, cfg, in_dim, K, proto_dim, data):
        super().__init__()

        # ---------- 配置 ----------
        self.use_adaptive_assign = getattr(cfg, 'use_adaptive_assign', True)
        self.tau_base = getattr(cfg, 'assign_tau_base', 0.3)      # 初始温度（较软）
        self.tau_decay = getattr(cfg, 'assign_tau_decay', 0.6)    # 每层衰减率
        self.K = K
        self.cfg = cfg
        self.feedback_query = nn.Linear(proto_dim, in_dim)

        # ---------- 层维度 ----------
        dims = [in_dim] + [cfg.out_dim] * (cfg.pc_num_layers-1) + [cfg.out_dim]

        # ---------- 卷积层 ----------
        self.layers = nn.ModuleList()
        for i in range(cfg.pc_num_layers):
            self.layers.append(
                ProtoConditionedConv(
                    in_dim=dims[i], out_dim=dims[i + 1], K=K, proto_dim=proto_dim,
                    use_hypernet=cfg.use_hypernet, hyper_hidden=cfg.hyper_hidden,
                    attn_mode=cfg.pc_attn_mode
                )
            )



        # # ---------- 层间分配演化（除最后一层外）----------
        # if self.use_adaptive_assign:
        #     self.assign_adapters = nn.ModuleList([
        #         AssignmentAdapter(dims[i+1], K, step_size=getattr(cfg, 'assign_step', 0.1))
        #         for i in range(cfg.pc_num_layers-1)
        #     ])

        # ---------- 输出投影与残差 ----------
        self.proj_out = nn.Linear(dims[-1], cfg.out_dim)
        self.residual = cfg.pc_residual
        self.proj_x = nn.Linear(data.x.shape[1], dims[0])
        if self.residual:
            self.res_proj = nn.Linear(in_dim, cfg.out_dim)

    def forward(self, h, edge_index, A_batch, P, C, proto_rep):
        """
        输入：
            h: (N, in_dim) 初始节点表示（通常是 h_smooth）
            edge_index: (2, E)
            A_batch: (N, K) 初始分配矩阵（来自 Sinkhorn）
            P, C, proto_rep: 原型相关

        返回：
            h: (N, out_dim) 最终节点表示
            A_list: list of (N, K) 每层的分配矩阵（用于对齐、解释、忠实度测试）
        """
        h_orig = h
        A = A_batch
        A_list = []                                        # 记录初始 A^(0)
        hs = []
        if h_orig.shape[1]!=P.shape[1]:
            h_orig = self.proj_x(h_orig)
        # ---------- 逐层传播 ----------
        h = self.feedback_query(torch.mm(A_batch, P))

        if len(self.layers) == 0:
            hs.append(h)
            r_vk =  h @ P.t()
            A_list.append(F.softmax(r_vk, dim=-1))
        else:
            for i, conv in enumerate(self.layers):
                # # 层间 A 演化（除最后一层）
                # if self.use_adaptive_assign and 0< i < len(self.layers):
                #     # tau 随层深递减：越深层，分配越硬
                #     tau = self.tau_base * (self.tau_decay ** (i-1))
                #     A = self.assign_adapters[i-1](h, A, tau)
                #     A_list.append(A)
                # 用当前 A 做条件聚合
                h,r_vk = conv(h, edge_index, A, P, C, proto_rep)
                h = F.relu(h)
                hs.append(h)
                A_list.append(r_vk)




        # h_proto = h

        # # 维度对齐：如果 proto_dim ≠ out_dim，先投影
        # if P.size(1) != self.cfg.out_dim:
        #     h = self.proj_out(h_proto)
        # else:
        #     # 即使维度匹配，也过一下 proj_out（可学习融合）
        #     h = self.proj_out(h_proto)

        # 残差连接
        if self.residual:
            h = h + self.res_proj(h_orig)

        return h, A_list,hs

