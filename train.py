import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack


"""
LeWM (Latent World Model) 训练脚本

该脚本实现了基于 JEPA (Joint Embedding Predictive Architecture) 的世界模型训练。
主要功能：
1. 加载和处理 HDF5 格式的数据集
2. 构建编码器、预测器和动作编码器等模型组件
3. 训练世界模型并保存模型权重
4. 支持 Weights & Biases 日志记录
"""


def lejepa_forward(self, batch, stage, cfg):
    """
    LeWM 模型的前向传播函数
    
    该函数执行以下操作：
    1. 编码观测数据
    2. 基于历史上下文预测未来状态
    3. 计算预测损失和正则化损失
    
    参数:
        self: 模型模块实例
        batch: 输入批次数据，包含观测和动作
        stage: 训练阶段标识 ('train' 或 'val')
        cfg: 配置对象
        
    返回:
        output: 包含嵌入向量、损失值等信息的字典
    """
    
    # 获取配置参数
    ctx_len = cfg.wm.history_size      # 历史上下文长度
    n_preds = cfg.wm.num_preds         # 预测步数
    lambd = cfg.loss.sigreg.weight     # SIGReg 正则化损失权重

    # 将 NaN 值替换为 0（在序列边界处会出现 NaN）
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # 编码批次数据，获取观测嵌入和动作嵌入
    output = self.model.encode(batch)

    # 提取嵌入向量
    # emb: 观测嵌入，形状为 (B, T, D)，其中 B=批次大小, T=时间步数, D=嵌入维度
    emb = output["emb"]
    act_emb = output["act_emb"]        # 动作嵌入

    # 提取历史上下文
    ctx_emb = emb[:, :ctx_len]         # 历史观测嵌入
    ctx_act = act_emb[:, : ctx_len]    # 历史动作嵌入

    # 准备目标嵌入和预测嵌入
    tgt_emb = emb[:, n_preds:]         # 目标嵌入（真实未来状态）
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # 预测的未来状态嵌入

    # 计算 LeWM 损失
    # 预测损失：预测嵌入与目标嵌入之间的均方误差
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    
    # SIGReg 正则化损失：防止嵌入空间坍塌
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    
    # 总损失 = 预测损失 + λ * 正则化损失
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    # 记录损失值到日志
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    """
    主训练函数
    
    该函数使用 Hydra 进行配置管理，执行完整的训练流程：
    1. 数据集加载和预处理
    2. 模型组件构建
    3. 训练器配置
    4. 模型训练和保存
    
    参数:
        cfg: Hydra 配置对象，包含所有训练参数
    """
    
    #########################
    ##       数据集准备      ##
    #########################

    # 加载 HDF5 格式的数据集
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    
    # 初始化数据变换列表，首先添加图像预处理器
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    # 为非图像数据列添加归一化器
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            # 跳过像素数据列
            if col.startswith("pixels"):
                continue

            # 为该列创建归一化器并添加到变换列表
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            # 将数据维度保存到配置中，供模型使用
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    # 组合所有变换并应用到数据集
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # 创建随机数生成器并设置随机种子，确保可重复性
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    
    # 将数据集划分为训练集和验证集
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # 创建数据加载器
    # 训练集：打乱数据，丢弃最后不完整的批次
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen)
    # 验证集：不打乱数据，保留所有样本
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       模型与优化器构建     ##
    ##############################

    # 创建 Vision Transformer 编码器
    # 使用 HuggingFace 预训练模型架构，但不加载预训练权重
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,           # 模型规模（如 'base', 'large'）
        patch_size=cfg.patch_size,   # 图像块大小
        image_size=cfg.img_size,     # 输入图像尺寸
        pretrained=False,            # 不使用预训练权重
        use_mask_token=False,        # 不使用掩码标记
    )

    # 获取模型维度参数
    hidden_dim = encoder.config.hidden_size                    # 编码器隐藏层维度
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)           # 嵌入维度（默认与隐藏层相同）
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim  # 有效动作维度（考虑帧跳跃）

    # 创建自回归预测器
    # 用于基于历史上下文预测未来状态
    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,  # 历史帧数
        input_dim=embed_dim,              # 输入嵌入维度
        hidden_dim=hidden_dim,            # 隐藏层维度
        output_dim=hidden_dim,            # 输出维度
        **cfg.predictor,                  # 其他预测器配置参数
    )

    # 创建动作编码器
    # 将动作向量映射到嵌入空间
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    # 创建投影器
    # 将编码器输出投影到嵌入空间
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,                  # MLP 隐藏层维度
        norm_fn=torch.nn.BatchNorm1d,     # 使用批归一化
    )

    # 创建预测器投影器
    # 将预测器输出投影到嵌入空间
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    # 组装完整的 JEPA 世界模型
    world_model = JEPA(
        encoder=encoder,                  # 视觉编码器
        predictor=predictor,              # 自回归预测器
        action_encoder=action_encoder,    # 动作编码器
        projector=projector,              # 编码器投影器
        pred_proj=predictor_proj,         # 预测器投影器
    )

    # 配置优化器和学习率调度器
    optimizers = {
        'model_opt': {
            "modules": 'model',           # 优化目标模块
            "optimizer": dict(cfg.optimizer),  # 优化器参数
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},  # 学习率调度器：线性预热 + 余弦退火
            "interval": "epoch",          # 每个 epoch 更新学习率
        },
    }

    # 创建数据模块
    data_module = spt.data.DataModule(train=train, val=val)
    
    # 创建训练模块
    # 将模型、损失函数、前向传播函数和优化器封装在一起
    world_model = spt.Module(
        model=world_model,                # 世界模型
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),  # SIGReg 正则化模块
        forward=partial(lejepa_forward, cfg=cfg),  # 前向传播函数
        optim=optimizers,                 # 优化器配置
    )

    ##########################
    ##       训练配置        ##
    ##########################

    # 设置运行目录
    run_id = cfg.get("subdir") or ""      # 获取运行 ID（子目录名）
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)  # 构建完整运行目录路径

    # 配置 Weights & Biases 日志记录器
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)  # 创建 WandB 日志记录器
        logger.log_hyperparams(OmegaConf.to_container(cfg))  # 记录超参数

    # 创建运行目录并保存配置文件
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)  # 保存配置到 YAML 文件

    # 创建模型保存回调
    # 每个 epoch 结束后保存完整的模型对象
    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,                      # 保存目录
        filename=cfg.output_model_name,       # 模型文件名
        epoch_interval=1,                     # 每个 epoch 保存一次
    )

    # 创建 PyTorch Lightning 训练器
    trainer = pl.Trainer(
        **cfg.trainer,                        # 训练器配置参数
        callbacks=[object_dump_callback],     # 回调函数列表
        num_sanity_val_steps=1,               # 训练前验证步数（用于检查代码正确性）
        logger=logger,                        # 日志记录器
        enable_checkpointing=True,            # 启用检查点保存
    )

    # 创建训练管理器
    # 管理训练过程，包括恢复训练、保存检查点等
    manager = spt.Manager(
        trainer=trainer,                      # 训练器
        module=world_model,                   # 训练模块
        data=data_module,                     # 数据模块
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",  # 检查点路径
    )

    # 启动训练
    manager()
    return


if __name__ == "__main__":
    run()
