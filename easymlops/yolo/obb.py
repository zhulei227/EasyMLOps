from easymlops.yolo.core.pipe import YOLOPipeObjectBase
from easymlops.table.core import dataframe_type, dict_type
import pandas as pd
import numpy as np


class YOLOOBB(YOLOPipeObjectBase):
    """
    YOLO旋转目标检测Pipe。
    
    基于Ultralytics YOLO模型进行旋转目标检测，支持检测任意角度的旋转目标。
    
    Example:
        >>> from easymlops.yolo.obb import YOLOOBB
        >>> pipe = YOLOOBB(model_name="yolo11n-obb")
        >>> results = pipe.transform(df)
    """
    
    def __init__(self, model_name="yolo11n-obb", model_path=None, conf=0.25, iou=0.45,
                 imgsz=640, device="cpu", save=False, show=False, **kwargs):
        """
        初始化旋转目标检测Pipe。
        
        Args:
            model_name: 模型名称，如 yolo11n-obb, yolo11s-obb 等
            model_path: 模型文件路径，如果指定则优先使用
            conf: 置信度阈值
            iou: NMS IoU阈值
            imgsz: 输入图像尺寸
            device: 运行设备，auto/cpu/cuda
            save: 是否保存结果
            show: 是否显示结果
            **kwargs: 其他父类参数
        """
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            task_type="obb",
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save=save,
            show=show,
            **kwargs
        )

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
        批量旋转目标检测。
        
        Args:
            s: 输入数据，DataFrame，包含image_path列
            
        Returns:
            包含旋转目标检测结果的DataFrame
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
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                save=self.save,
                show=self.show,
                verbose=False
            )[0]
            
            boxes = result.obb
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    result_dict = {
                        "image_path": image_path,
                        "class_id": int(box.cls[0]),
                        "class_name": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox_x1": float(box.xyxyxyxy[0][0]),
                        "bbox_y1": float(box.xyxyxyxy[0][1]),
                        "bbox_x2": float(box.xyxyxyxy[0][2]),
                        "bbox_y2": float(box.xyxyxyxy[0][3]),
                        "bbox_x3": float(box.xyxyxyxy[0][4]),
                        "bbox_y3": float(box.xyxyxyxy[0][5]),
                        "bbox_x4": float(box.xyxyxyxy[0][6]),
                        "bbox_y4": float(box.xyxyxyxy[0][7]),
                        "rotation": float(box.xywhr[0][2]) if hasattr(box, 'xywhr') else None,
                        "angle": float(box.xywhr[0][2]) if hasattr(box, 'xywhr') else None
                    }
                    results_list.append(result_dict)
            else:
                results_list.append({
                    "image_path": image_path,
                    "class_id": None,
                    "class_name": None,
                    "confidence": None,
                    "bbox_x1": None,
                    "bbox_y1": None,
                    "bbox_x2": None,
                    "bbox_y2": None,
                    "bbox_x3": None,
                    "bbox_y3": None,
                    "bbox_x4": None,
                    "bbox_y4": None,
                    "rotation": None,
                    "angle": None
                })
        
        return pd.DataFrame(results_list)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单张图像旋转目标检测。
        
        Args:
            s: 输入数据，字典，包含image_path键
            
        Returns:
            包含旋转目标检测结果的字典
        """
        from ultralytics import YOLO
        
        image_path = s.get("image_path", s.get("path", s.get("image", None)))
        if image_path is None:
            raise ValueError("未找到图像路径")
        
        result = self.model.predict(
            image_path,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            save=self.save,
            show=self.show,
            verbose=False
        )[0]
        
        boxes = result.obb
        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxyxyxy = box.xyxyxyxy[0].cpu().numpy().tolist()
                rotation = float(box.xywhr[0][2]) if hasattr(box, 'xywhr') else None
                
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "polygon": xyxyxyxy,
                    "rotation": rotation,
                    "angle": rotation
                })
        
        s["obb_detections"] = detections
        return s
