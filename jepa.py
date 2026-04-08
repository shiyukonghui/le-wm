"""JEPA (Joint Embedding Predictive Architecture) 实现"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    """分离并克隆张量，如果不是张量则直接返回
    
    Args:
        v: 输入值，可以是张量或其他类型
        
    Returns:
        如果是张量则返回分离并克隆的副本，否则返回原值
    """
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):
    """JEPA 模型：联合嵌入预测架构
    
    用于学习状态表示和预测未来状态的模型，包含编码器、预测器和动作编码器。
    """
    
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        """初始化 JEPA 模型
        
        Args:
            encoder: 观测编码器，将像素观测转换为嵌入表示
            predictor: 预测器，根据当前嵌入和动作预测未来嵌入
            action_encoder: 动作编码器，将动作转换为嵌入表示
            projector: 可选的投影层，用于投影观测嵌入，默认为恒等映射
            pred_proj: 可选的预测投影层，用于投影预测结果，默认为恒等映射
        """
        super().__init__()
        
        self.encoder = encoder  # 观测编码器
        self.predictor = predictor  # 状态预测器
        self.action_encoder = action_encoder  # 动作编码器
        self.projector = projector or nn.Identity()  # 观测嵌入投影层
        self.pred_proj = pred_proj or nn.Identity()  # 预测结果投影层

    def encode(self, info):
        """编码观测和动作为嵌入表示
        
        Args:
            info: 包含 'pixels' 和可选 'action' 键的字典
                - pixels: 像素观测张量，形状为 (B, T, ...)
                - action: 动作张量（可选）
        
        Returns:
            更新后的 info 字典，包含:
                - emb: 观测嵌入，形状为 (B, T, D)
                - act_emb: 动作嵌入（如果提供了动作）
        """
        
        pixels = info['pixels'].float()  # 获取像素观测并转换为浮点型
        b = pixels.size(0)  # 批次大小
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # 展平批次和时间维度以便编码
        output = self.encoder(pixels, interpolate_pos_encoding=True)  # 编码像素观测
        pixels_emb = output.last_hidden_state[:, 0]  # 提取 [CLS] token 作为嵌入
        emb = self.projector(pixels_emb)  # 投影嵌入
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)  # 恢复批次和时间维度
        
        # 如果存在动作，则编码动作
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        
        return info

    def predict(self, emb, act_emb):
        """预测下一个状态嵌入
        
        Args:
            emb: 当前状态嵌入，形状为 (B, T, D)
            act_emb: 动作嵌入，形状为 (B, T, A_emb)
        
        Returns:
            预测的状态嵌入，形状为 (B, T, D)
        """
        preds = self.predictor(emb, act_emb)  # 使用预测器预测未来状态
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))  # 投影预测结果
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))  # 恢复形状
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """根据初始信息字典和动作序列展开模型预测
        
        Args:
            info: 包含初始状态信息的字典
                - pixels: 初始像素观测，形状为 (B, S, T, C, H, W)
            action_sequence: 动作序列，形状为 (B, S, T, action_dim)
                - S: 动作计划样本数
                - T: 时间范围
            history_size: 用于预测的历史窗口大小，默认为 3
        
        Returns:
            更新后的 info 字典，包含:
                - predicted_emb: 预测的嵌入序列，形状为 (B, S, T, D)
        """
        
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)  # 历史长度
        B, S, T = action_sequence.shape[:3]  # 批次大小、样本数、时间范围
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)  # 分离初始动作和未来动作
        info["action"] = act_0
        n_steps = T - H  # 需要预测的步数
        
        # 复制并编码初始信息字典
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)  # 扩展到 S 个样本
        _init = {k: detach_clone(v) for k, v in _init.items()}
        
        # 展平批次和样本维度以便展开
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")
        
        # 自回归地展开预测器 n_steps 步
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)  # 编码动作
            emb_trunc = emb[:, -HS:]  # 截取历史窗口 (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # 截取历史动作 (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # 预测下一步 (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # 拼接预测结果 (BS, T+1, D)
            
            next_act = act_future[:, t : t + 1, :]  # 获取下一个动作 (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # 拼接动作 (BS, T+1, action_dim)
        
        # 预测最后状态
        act_emb = self.action_encoder(act)  # 编码所有动作 (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # 截取历史窗口 (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # 截取历史动作 (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # 预测最后状态 (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)
        
        # 恢复批次和样本维度
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        
        return info

    def criterion(self, info_dict: dict):
        """计算预测嵌入和目标嵌入之间的损失
        
        Args:
            info_dict: 包含预测和目标信息的字典
                - predicted_emb: 预测的嵌入，形状为 (B, S, T-1, dim)
                - goal_emb: 目标嵌入，形状为 (B, S, T, dim)
        
        Returns:
            每个动作候选的最后一步损失，形状为 (B, S)
        """
        pred_emb = info_dict["predicted_emb"]  # 预测的嵌入 (B, S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # 目标嵌入 (B, S, T, dim)
        
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)  # 扩展目标嵌入以匹配预测形状
        
        # 返回每个动作候选的最后一步损失
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)
        
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """计算给定目标状态和初始状态下动作候选的成本
        
        Args:
            info_dict: 包含目标和初始状态信息的字典
                - goal: 目标状态像素观测
                - pixels: 初始状态像素观测
            action_candidates: 动作候选序列，形状为 (B, S, T, action_dim)
        
        Returns:
            每个动作候选的成本，形状为 (B, S)
        """
        
        assert "goal" in info_dict, "goal not in info_dict"
        
        device = next(self.parameters()).device
        # 将所有张量移动到模型所在设备
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        
        # 提取并处理目标状态
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        
        # 重命名目标相关的键
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)
        
        goal.pop("action")  # 移除动作键
        goal = self.encode(goal)  # 编码目标状态
        
        info_dict["goal_emb"] = goal["emb"]  # 保存目标嵌入
        info_dict = self.rollout(info_dict, action_candidates)  # 展开预测
        
        cost = self.criterion(info_dict)  # 计算成本
        
        return cost
