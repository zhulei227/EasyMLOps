from easymlops.yolo.core.pipe import YOLOPipeObjectBase
from easymlops.table.core import dataframe_type, dict_type
import pandas as pd
import numpy as np


class YOLOClassification(YOLOPipeObjectBase):
    """
    YOLO图像分类Pipe。
    
    基于Ultralytics YOLO模型进行图像分类，支持对图像进行多类别分类。
    
    Example:
        >>> from easymlops.yolo.classification import YOLOClassification
        >>> pipe = YOLOClassification(model_name="yolo11n-cls")
        >>> results = pipe.transform(df)
    """
    
    def __init__(self, model_name="yolo11n-cls", model_path=None, conf=0.25,
                 imgsz=224, device="cpu", save=False, show=False, **kwargs):
        """
        初始化图像分类Pipe。
        
        Args:
            model_name: 模型名称，如 yolo11n-cls, yolo11s-cls 等
            model_path: 模型文件路径，如果指定则优先使用
            conf: 置信度阈值
            imgsz: 输入图像尺寸
            device: 运行设备，auto/cpu/cuda
            save: 是否保存结果
            show: 是否显示结果
            **kwargs: 其他父类参数
        """
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            task_type="classify",
            conf=conf,
            imgsz=imgsz,
            device=device,
            save=save,
            show=show,
            **kwargs
        )
        self.top_k = 5

    def udf_fit(self, s: dataframe_type, **kwargs):
        """加载模型。"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics-opencv-headless")
        
        if self.model_path:
            self.model = YOLO(self.model_path)
        else:
            model_file = f"{self.model_name}.pt"
            self.model = YOLO(model_file)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量图像分类。
        
        Args:
            s: 输入数据，DataFrame，包含image_path列
            
        Returns:
            包含分类结果的DataFrame
        """
        from ultralytics import YOLO
        
        results_list = []
        for idx, row in s.iterrows():
            image_path = row.get("image_path", row.get("path", row.get("image", None)))
            if image_path is None:
                raise ValueError("未找到图像路径列，请确保数据包含image_path/path/image列")
            
            result = self.model.predict(
                image_path,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                save=self.save,
                show=self.show,
                verbose=False
            )[0]
            
            probs = result.probs
            if probs is not None:
                top5_indices = probs.top5
                top5_conf = probs.top5conf.cpu().numpy().tolist()
                top5_names = [result.names[i] for i in top5_indices]
                
                result_dict = {
                    "image_path": image_path,
                    "top1_class": top5_names[0],
                    "top1_conf": top5_conf[0],
                    "top5_classes": top5_names,
                    "top5_confs": top5_conf
                }
            else:
                result_dict = {
                    "image_path": image_path,
                    "top1_class": None,
                    "top1_conf": None,
                    "top5_classes": None,
                    "top5_confs": None
                }
            
            results_list.append(result_dict)
        
        return pd.DataFrame(results_list)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单张图像分类。
        
        Args:
            s: 输入数据，字典，包含image_path键
            
        Returns:
            包含分类结果的字典
        """
        from ultralytics import YOLO
        
        image_path = s.get("image_path", s.get("path", s.get("image", None)))
        if image_path is None:
            raise ValueError("未找到图像路径")
        
        result = self.model.predict(
            image_path,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            save=self.save,
            show=self.show,
            verbose=False
        )[0]
        
        probs = result.probs
        if probs is not None:
            top5_indices = probs.top5
            top5_conf = probs.top5conf.cpu().numpy().tolist()
            top5_names = [result.names[i] for i in top5_indices]
            
            s["top1_class"] = top5_names[0]
            s["top1_conf"] = top5_conf[0]
            s["top5_classes"] = top5_names
            s["top5_confs"] = top5_conf
        else:
            s["top1_class"] = None
            s["top1_conf"] = None
            s["top5_classes"] = None
            s["top5_confs"] = None
        
        return s
