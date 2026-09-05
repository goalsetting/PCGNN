# sinkhorn.py
import torch


def sinkhorn_ot(cost, epsilon, llm_prior=None, llm_lambda=0.0, max_iter=5, tol=1e-6):
    """
    求解带有 KL 先验的熵正则化最优传输的分配矩阵。
    Args:
        cost: (B, K) 代价矩阵，B=mini-batch 节点数，K=原型数。
        epsilon: 熵正则系数。
        llm_prior: (B, K) LLM 给出的参考分布，无先验行应为均匀分布 (1/K)。
                    如果为 None，则无 LLM 先验。
        llm_lambda: 控制 LLM 先验强度。
        max_iter: Sinkhorn 最大迭代次数。
        tol: 停止阈值。
    Returns:
        A: (B, K) 软分配矩阵，每行和为 1/B，每列和为 1/K。
    """
    B, K = cost.shape
    device = cost.device
    tol = 1e-6
    # 融合 LLM 先验: C' = C - lambda * log(R)
    if llm_prior is not None:
        # 避免 log(0)，添加极小值
        log_prior = torch.log(llm_prior + 1e-10)
        cost_mod = cost - llm_lambda * log_prior
    else:
        cost_mod = cost

    # 有效熵正则系数，需确保大于0
    epsilon_eff = max(epsilon - llm_lambda, 1e-3)
    #
    # # 初始化对数边际
    # log_a = -torch.log(torch.tensor(B, device=device))  # log(1/B)
    # log_b = -torch.log(torch.tensor(K, device=device))  # log(1/K)
    #
    # # 初始化 u, v 的对数
    # log_u = torch.zeros(B, 1, device=device)
    # log_v = torch.zeros(K, 1, device=device)
    #
    # # 使用 log-domain 迭代
    # for _ in range(max_iter):
    #     log_u_prev = log_u
    #     log_v_prev = log_v
    #
    #     # 更新 log_v: log_v = log_b - logsumexp( log_u + (-cost_mod/epsilon_eff) , dim=0 )
    #     # 这里利用 logsumexp 技巧
    #     log_K = -cost_mod / epsilon_eff  # 仍然可能很大，但不会计算 exp
    #     # log_v = log_b - logsumexp( log_u + log_K, dim=0 )
    #     temp1 = log_u + log_K  # (B, K)
    #     log_v = log_b - torch.logsumexp(temp1, dim=0, keepdim=True)  # (1, K)
    #
    #     # 更新 log_u: log_u = log_a - logsumexp( log_K + log_v, dim=1 )
    #     temp2 = log_K + log_v  # (B, K)
    #     log_u = log_a - torch.logsumexp(temp2, dim=1, keepdim=True)  # (B, 1)
    #
    #     # 收敛检查（可在对数域检查变化量）
    #     if torch.max(torch.abs(log_u - log_u_prev)) < tol and torch.max(torch.abs(log_v - log_v_prev)) < tol:
    #         break
    #
    # # 从对数域恢复分配矩阵：A = exp(log_u + log_K + log_v)
    # log_A = log_u + log_K + log_v  # (B, K)
    # A = torch.exp(log_A)  # 这里 exp 的参数通常已经是安全范围内的负数或小值，因为 logsumexp 已稳定
    # return A

    # 计算核矩阵 K = exp(-C'/epsilon_eff)
    K_m = torch.exp(-cost_mod / epsilon_eff)  # (B, K)

    # 目标行/列和
    r = torch.ones(B, 1, device=device) / B
    c = torch.ones(K, 1, device=device) / K

    u = torch.ones(B, 1, device=device)
    v = torch.ones(K, 1, device=device)

    for _ in range(max_iter):
        u_prev = u
        v_prev = v
        # 更新 u
        u = r / (torch.mm(K_m, v) + 1e-10)
        # 更新 v
        v = c / (torch.mm(K_m.t(), u) + 1e-10)

        # 提前停止检查
        if torch.max(torch.abs(u - u_prev)) < tol and torch.max(torch.abs(v - v_prev)) < tol:
            break

    # 闭式解 A = diag(u) * K * diag(v)
    A = u * K_m * v.t()  # 广播乘法


    return A