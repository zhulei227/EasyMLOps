from easymlops.table.core import TablePipeObjectBase
import pandas as pd
import numpy as np

dataframe_type = pd.DataFrame
dict_type = dict


class YOLOPipeObjectBase(TablePipeObjectBase):
    """
    YOLO Pipe基类。
    
    继承自TablePipeObjectBase，专门用于处理YOLO相关任务。
    支持目标检测、实例分割、图像分类、姿态估计、旋转目标检测等任务。
    
    Example:
        >>> class MyYOLOPipe(YOLOPipeObjectBase):
        ...     def udf_transform(self, s, **kwargs):
        ...         return s
    """
    
    def __init__(self, model_name=None, model_path=None, task_type="detect",
                 conf=0.25, iou=0.45, imgsz=640, device="cpu",
                 save=False, show=False, **kwargs):
        """
        初始化YOLO Pipe对象。
        
        Args:
            model_name: 模型名称，如 yolo11n, yolo11s 等
            model_path: 模型文件路径，如果指定则优先使用
            task_type: 任务类型，detect/segment/classify/pose/obb
            conf: 置信度阈值
            iou: NMS IoU阈值
            imgsz: 输入图像尺寸
            device: 运行设备，auto/cpu/cuda
            save: 是否保存结果
            show: 是否显示结果
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_path = model_path
        self.task_type = task_type
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.save = save
        self.show = show
        self.model = None

    def udf_get_params(self):
        """获取参数。"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "task_type": self.task_type,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "device": self.device,
            "save": self.save,
            "show": self.show
        }

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        self.model_name = params.get("model_name", self.model_name)
        self.model_path = params.get("model_path", self.model_path)
        self.task_type = params.get("task_type", self.task_type)
        self.conf = params.get("conf", self.conf)
        self.iou = params.get("iou", self.iou)
        self.imgsz = params.get("imgsz", self.imgsz)
        self.device = params.get("device", self.device)
        self.save = params.get("save", self.save)
        self.show = params.get("show", self.show)
