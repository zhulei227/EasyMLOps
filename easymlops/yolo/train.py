from easymlops.yolo.core.pipe import YOLOPipeObjectBase
from easymlops.table.core import dataframe_type, dict_type
import pandas as pd
import numpy as np


class YOLOTrain(YOLOPipeObjectBase):
    """
    YOLO模型训练Pipe。
    
    基于Ultralytics YOLO模型进行训练，支持目标检测、实例分割等任务的模型训练。
    
    Example:
        >>> from easymlops.yolo.train import YOLOTrain
        >>> pipe = YOLOTrain(data="dataset.yaml", epochs=100)
        >>> pipe.fit(df)
    """
    
    def __init__(self, model_name="yolo11n", data=None, epochs=100, imgsz=640,
                 batch=16, device="cpu", workers=0, **kwargs):
        """
        初始化训练Pipe。
        
        Args:
            model_name: 预训练模型名称，如 yolo11n, yolo11s 等
            data: 数据集配置文件路径 (dataset.yaml)
            epochs: 训练轮数
            imgsz: 输入图像尺寸
            batch: 批次大小
            device: 运行设备，auto/cpu/cuda
            workers: 数据加载线程数
            **kwargs: 其他父类参数
        """
        super().__init__(
            model_name=model_name,
            task_type="train",
            **kwargs
        )
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.workers = workers

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        训练模型。
        
        Args:
            s: 输入数据（DataFrame，用于验证或预留）
            **kwargs: 其他训练参数
            
        Returns:
            self
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics-opencv-headless")
        
        if not self.data:
            raise ValueError("请指定数据集配置文件路径 (data 参数)")
        
        model_file = f"{self.model_name}.pt"
        self.model = YOLO(model_file)
        
        self.model.train(
            data=self.data,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            workers=self.workers,
            **kwargs
        )
        
        return self


class YOLOVal(YOLOPipeObjectBase):
    """
    YOLO模型验证Pipe。
    
    基于Ultralytics YOLO模型进行验证，评估模型性能。
    
    Example:
        >>> from easymlops.yolo.train import YOLOVal
        >>> pipe = YOLOVal(model_path="best.pt", data="dataset.yaml")
        >>> metrics = pipe.transform(df)
    """
    
    def __init__(self, model_path=None, data=None, imgsz=640,
                 batch=16, device="cpu", **kwargs):
        """
        初始化验证Pipe。
        
        Args:
            model_path: 模型文件路径
            data: 数据集配置文件路径
            imgsz: 输入图像尺寸
            batch: 批次大小
            device: 运行设备，auto/cpu/cuda
            **kwargs: 其他父类参数
        """
        super().__init__(
            model_path=model_path,
            task_type="val",
            imgsz=imgsz,
            device=device,
            **kwargs
        )
        self.data = data
        self.batch = batch

    def udf_fit(self, s: dataframe_type, **kwargs):
        """加载模型。"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics")
        
        if self.model_path:
            self.model = YOLO(self.model_path)
        else:
            raise ValueError("请指定模型路径 (model_path 参数)")
        
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        验证模型。
        
        Args:
            s: 输入数据（DataFrame）
            
        Returns:
            包含验证指标的DataFrame
        """
        metrics = self.model.val(
            data=self.data,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            **kwargs
        )
        
        results = {
            "map50": getattr(metrics, 'map50', None),
            "map50_95": getattr(metrics, 'map', None),
            "precision": getattr(metrics, 'mp', None),
            "recall": getattr(metrics, 'mr', None)
        }
        
        return pd.DataFrame([results])


class YOLOExport(YOLOPipeObjectBase):
    """
    YOLO模型导出Pipe。
    
    基于Ultralytics YOLO模型进行格式转换，支持导出为ONNX, CoreML, TensorFlow Lite等格式。
    
    Example:
        >>> from easymlops.yolo.train import YOLOExport
        >>> pipe = YOLOExport(model_path="best.pt", format="onnx")
        >>> pipe.transform(df)
    """
    
    def __init__(self, model_path=None, format="onnx", imgsz=640,
                 dynamic=False, simplify=True, **kwargs):
        """
        初始化导出Pipe。
        
        Args:
            model_path: 模型文件路径
            format: 导出格式，onnx/coreml/tflite/torchscript/engine
            imgsz: 输入图像尺寸
            dynamic: 是否支持动态输入尺寸
            simplify: 是否简化ONNX模型
            **kwargs: 其他父类参数
        """
        super().__init__(
            model_path=model_path,
            task_type="export",
            imgsz=imgsz,
            **kwargs
        )
        self.format = format
        self.dynamic = dynamic
        self.simplify = simplify

    def udf_fit(self, s: dataframe_type, **kwargs):
        """加载模型。"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics")
        
        if self.model_path:
            self.model = YOLO(self.model_path)
        else:
            raise ValueError("请指定模型路径 (model_path 参数)")
        
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        导出模型。
        
        Args:
            s: 输入数据（DataFrame，预留）
            
        Returns:
            包含导出路径的DataFrame
        """
        export_path = self.model.export(
            format=self.format,
            imgsz=self.imgsz,
            dynamic=self.dynamic,
            simplify=self.simplify,
            **kwargs
        )
        
        return pd.DataFrame([{"export_path": export_path}])
