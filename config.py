# config.py
import io

import yaml


class Config:
    def __init__(self, yaml_path):
        # with open(yaml_path, 'r') as f:
        #     cfg = yaml.safe_load(f)
        with io.open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 图与嵌入
        self.in_feats = cfg['model']['in_feats']
        self.hidden_dim = cfg['model']['hidden_dim']
        self.out_dim = cfg['model']['out_dim']  # 节点嵌入维度
        self.num_layers = cfg['model']['num_layers']  # GNN 层数

        # 原型
        self.num_prototypes = cfg['prototype']['num_prototypes']
        self.context_hidden = cfg['prototype']['context_hidden']  # 上下文 MLP 隐层维度
        self.tau = cfg['prototype']['temperature']  # 用于对比损失的 τ
        self.tau_cert = cfg['prototype']['tau_cert']
        self.expl_strategy = cfg['prototype']['expl_strategy']
        self.expl_k = cfg['prototype']['expl_k']
        self.expl_tau = cfg['prototype']['expl_tau']

        # Sinkhorn 最优传输
        self.ot_epsilon = cfg['ot']['epsilon']  # 熵正则系数
        self.ot_max_iter = cfg['ot']['max_iter']  # Sinkhorn 最大迭代次数
        self.ot_tol = cfg['ot']['tol']  # 收敛容忍度

        # 图扩散
        self.diffusion_alpha = cfg['graph']['diffusion_alpha']  # 图平滑系数

        # LLM 先验
        self.llm_lambda = cfg['llm']['lambda']  # LLM 惩罚强度

        # 损失权重
        self.w_centroid = cfg['loss']['w_centroid']
        self.w_context = cfg['loss']['w_context']
        self.w_contrast = cfg['loss']['w_contrast']
        self.w_task = cfg['loss']['w_task']
        self.w_suff = cfg['loss']['w_suff']
        self.w_adapt_suff = cfg['loss']['w_adapt_suff']
        self.w_nec = cfg['loss']['w_nec']
        self.w_mask_align = cfg['loss']['w_mask_align']
        self.w_sparse = cfg['loss']['w_sparse']
        self.w_gate_entropy = cfg['loss']['w_gate_entropy']

        # 优化器
        self.lr = cfg['optim']['lr']
        self.weight_decay = cfg['optim']['weight_decay']

        # 设备
        self.device = cfg['device']  # 'cuda' 或 'cuda:0'

        # 新增概念图配置
        cg = cfg['concept_graph']
        self.knn_k = cg['knn_k']
        self.cg_num_layers = cg['num_layers']
        self.cg_dropout = cg['dropout']
        self.use_struct_sim = cg['use_struct']
        self.use_context_sim = cg['use_context']
        self.use_semantic_sim = cg['use_semantic']
        self.cg_fusion_mode = cg['fusion_mode']
        self.cg_adj_mode = cg['cg_adj_mode']
        self.cg_fusion_mode = cg['cg_fusion_mode']
        # 新增原型条件化 GNN 配置
        pcg = cfg['conditioned_gnn']
        self.pc_num_layers = pcg['num_layers']
        self.pc_hidden_dim = pcg['hidden_dim']
        self.pc_out_dim = pcg['out_dim']
        self.use_hypernet = pcg['use_hypernet']
        self.hyper_hidden = pcg['hyper_hidden']
        self.pc_residual = pcg['residual']
        self.pc_attn_mode = pcg['pc_attn_mode']