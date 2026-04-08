import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def img_transform(cfg):
    """图像预处理变换函数
    
    创建图像预处理流水线，包括转换为图像格式、数据类型转换、
    归一化和尺寸调整等操作。
    
    参数:
        cfg: 配置对象，包含图像尺寸等参数
        
    返回:
        transform: 组合后的图像变换操作
    """
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    """获取指定回合的长度
    
    根据回合索引和步数索引计算每个回合的长度（步数）。
    
    参数:
        dataset: 数据集对象
        episodes: 回合索引列表
        
    返回:
        np.array: 每个回合的长度数组
    """
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    """获取数据集
    
    从指定路径加载HDF5格式的数据集。
    
    参数:
        cfg: 配置对象，包含缓存目录等参数
        dataset_name: 数据集名称
        
    返回:
        dataset: 加载的HDF5数据集对象
    """
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """运行评估函数
    
    对世界模型策略与随机策略进行评估比较。
    使用Hydra框架进行配置管理。
    
    参数:
        cfg: Hydra配置字典对象
    """
    # 验证规划参数的合理性：规划视野与动作块的乘积不能超过评估预算
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # 设置世界模型的最大回合步数，设为评估预算的2倍以确保有足够空间
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    # 初始化世界模型，指定图像形状为224x224
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # 为像素和目标图像创建预处理变换器
    transform = {
        "pixels": img_transform(cfg),  # 当前观测图像的变换
        "goal": img_transform(cfg),    # 目标图像的变换
    }

    # 加载评估数据集
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    # 获取回合索引列名（兼容不同的数据集格式）
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    # 获取所有唯一的回合索引
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    # 为数据集中的各列创建标准化处理器
    process = {}
    for col in cfg.dataset.keys_to_cache:
        # 跳过像素列，因为像素数据通过transform处理
        if col in ["pixels"]:
            continue
        # 创建标准化缩放器并拟合数据
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        # 过滤掉包含NaN的数据行
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        # 为非动作列创建目标版本的处理期（用于目标条件）
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # 获取策略配置，默认为随机策略
    policy = cfg.get("policy", "random")

    if policy != "random":
        # 初始化自动代价模型用于评估
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")      # 将模型移至GPU
        model = model.eval()           # 设置为评估模式
        model.requires_grad_(False)    # 禁用梯度计算
        model.interpolate_pos_encoding = True  # 启用位置编码插值
        # 创建规划配置
        config = swm.PlanConfig(**cfg.plan_config)
        # 实例化求解器
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        # 创建世界模型策略
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        # 使用随机策略作为基线
        policy = swm.policy.RandomPolicy()

    # 确定结果保存路径
    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # 计算每个回合的长度
    episode_len = get_episodes_length(dataset, ep_indices)
    # 计算每个回合的最大起始索引（需要留出目标偏移步数）
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    # 创建回合ID到最大起始索引的映射字典
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    # 为数据集中的每一行计算其对应的最大起始索引
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # 创建有效起始点掩码：步数索引必须小于等于最大起始索引
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    # 使用配置的随机种子初始化随机数生成器
    g = np.random.default_rng(cfg.seed)
    # 从有效索引中随机选择评估起始点
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # 对选中的索引进行排序以保持顺序
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    # 获取选中行的回合索引和起始步数
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    # 验证是否有足够的有效回合进行评估
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    # 为世界模型设置策略
    world.set_policy(policy)

    # 执行评估并记录时间
    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),           # 起始步数列表
        goal_offset_steps=cfg.eval.goal_offset_steps,   # 目标偏移步数
        eval_budget=cfg.eval.eval_budget,               # 评估预算
        episodes_idx=eval_episodes.tolist(),            # 回合索引列表
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),  # 回调函数
        video_path=results_path,                        # 视频保存路径
    )
    end_time = time.time()

    print(metrics)

    # 构建结果文件路径并确保目录存在
    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # 将配置和结果写入文件
    with results_path.open("a") as f:
        f.write("\n")

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
