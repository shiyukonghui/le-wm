"""
工具函数模块
包含图像预处理、数据归一化和模型回调等功能
"""
import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """
    获取图像预处理器
    
    创建一个图像预处理管道，包含图像格式转换和尺寸调整
    
    参数:
        source: 源数据字段名
        target: 目标数据字段名
        img_size: 目标图像尺寸，默认为224
    
    返回:
        组合后的图像预处理变换器
    """
    # 获取ImageNet数据集的统计信息（均值和标准差）
    imagenet_stats = dt.dataset_stats.ImageNet
    
    # 创建图像格式转换变换器，使用ImageNet统计信息进行归一化
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    
    # 创建尺寸调整变换器
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    
    # 组合两个变换器，形成完整的预处理管道
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """
    获取指定列的归一化器
    
    根据数据集中指定列的数据计算均值和标准差，
    创建一个用于归一化该列数据的变换器
    
    参数:
        dataset: 数据集对象
        source: 源数据字段名
        target: 目标数据字段名
    
    返回:
        归一化变换器
    """
    # 从数据集中获取指定列的数据
    col_data = dataset.get_col_data(source)
    
    # 将numpy数组转换为PyTorch张量
    data = torch.from_numpy(np.array(col_data))
    
    # 移除包含NaN值的行，确保数据有效性
    data = data[~torch.isnan(data).any(dim=1)]
    
    # 计算数据的均值和标准差（按列计算）
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    # 定义归一化函数：标准化处理 (x - mean) / std
    def norm_fn(x):
        return ((x - mean) / std).float()

    # 创建包装后的变换器对象
    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


class ModelObjectCallBack(Callback):
    """
    模型对象保存回调类
    
    在每个训练epoch结束后保存完整的模型对象（包括结构和权重）
    继承自PyTorch Lightning的Callback基类
    """

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        """
        初始化回调对象
        
        参数:
            dirpath: 模型保存目录路径
            filename: 模型文件名前缀，默认为"model_object"
            epoch_interval: 保存间隔（每隔多少个epoch保存一次），默认为1
        """
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        """
        训练epoch结束时的回调方法
        
        根据设定的间隔保存模型对象，并在最后一个epoch强制保存
        
        参数:
            trainer: PyTorch Lightning训练器对象
            pl_module: PyTorch Lightning模块对象
        """
        super().on_train_epoch_end(trainer, pl_module)

        # 构建输出文件路径，包含epoch编号
        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        # 仅在主进程（global_zero）上执行保存操作，避免多GPU训练时重复保存
        if trainer.is_global_zero:
            # 按照设定的间隔保存模型
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # 在最后一个epoch强制保存模型
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        """
        保存模型对象到指定路径
        
        参数:
            model: 要保存的模型对象
            path: 保存路径
        """
        try:
            # 使用torch.save保存完整的模型对象
            torch.save(model, path)
        except Exception as e:
            # 捕获并打印保存过程中的异常
            print(f"Error saving model object: {e}")
