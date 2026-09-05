import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from evaluation import *

def kd_loss(student_logits, teacher_logits, T=2.0):
    """
    知识蒸馏损失：KL(教师软化概率 || 学生软化概率)
    Args:
        student_logits, teacher_logits: (N, C)
        T: 温度
    Returns:
        loss: 标量
    """
    loss = F.kl_div(student_logits, teacher_logits, reduction='batchmean')
    return loss


def train_node_classification_distill_start(model, data, optimizer, cfg, full_batch=True, epoch=0):
    model.train()
    optimizer.zero_grad()

    if not full_batch:
        raise NotImplementedError

    # ========== 1. 全原型前向（标准输出） ==========
    z_full, student_r_list, P_enh, C_enh, A_student, h_smooth = model.forward_student(
        data.x, data.edge_index, A_full=None, return_h_smooth=True,y= data.y, train_mask= data.train_mask
    )

    # ========== 2. 教师前向（蒸馏） ==========
    with torch.no_grad():
        teacher_final, teacher_r_list, ths = model.forward_teacher(data.x, data.edge_index)
        model.teacher_x = ths[-1]



    # # ========== 4. 原有对齐损失 ==========
    concept_loss = model.compute_concept_align_loss(student_r_list, teacher_r_list)

    total_loss = (
            cfg.w_contrast * concept_loss
    )

    total_loss.backward()
    optimizer.step()

    # 返回损失字典供日志记录
    loss_dict = {
        'total': total_loss.item(),
    }
    return loss_dict


def train_node_classification_distill(model, data, optimizer, cfg, full_batch=True, epoch=0):
    model.train()
    optimizer.zero_grad()

    if not full_batch:
        raise NotImplementedError

    # ========== 1. 全原型前向（标准输出） ==========
    z_full, student_r_list, P_enh, C_enh, A_student, h_smooth = model.forward_student(
        data.x, data.edge_index, A_full=None, return_h_smooth=True, y= data.y, train_mask= data.train_mask
    )

    # ========== 2. 教师前向（蒸馏） ==========
    with torch.no_grad():
        teacher_final, teacher_r_list,_ = model.forward_teacher(data.x, data.edge_index)
    kd = F.cross_entropy(z_full[data.train_mask], data.y[data.train_mask])


    # ========== 4. 原有对齐损失 ==========
    concept_loss = model.compute_concept_align_loss(student_r_list, teacher_r_list)



    # ========== 5. 新增：概念忠实度损失 ==========
    faith_criterion = ConceptFaithfulnessLoss(cfg)
    faith_losses = faith_criterion(model, data, z_full, A_student, h_smooth)

    # ========== 6. 总损失组装 ==========
    total_loss = (
            cfg.w_contrast * concept_loss +
            cfg.w_centroid * kd +
            cfg.w_suff * faith_losses['loss_suff'] +
            cfg.w_nec * faith_losses['loss_nec'] +
            cfg.w_mask_align * faith_losses['loss_mask_align'] +
            cfg.w_adapt_suff * faith_losses['loss_adapt_suff'] +
            cfg.w_sparse * faith_losses['loss_sparse'] +
            cfg.w_gate_entropy * faith_losses['loss_gate_entropy']
    )

    total_loss.backward()
    optimizer.step()

    # 返回损失字典供日志记录
    loss_dict = {
        'total': total_loss.item(),
        'concept_align': concept_loss.item(),
        'kd': kd.item(),
        'suff': faith_losses['loss_suff'].item(),
        'nec': faith_losses['loss_nec'].item(),
        'sparse': faith_losses['loss_sparse'].item(),
    }
    return loss_dict
