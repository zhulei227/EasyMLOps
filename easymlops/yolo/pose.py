from easymlops.yolo.core.pipe import YOLOPipeObjectBase
from easymlops.table.core import dataframe_type, dict_type
import pandas as pd
import numpy as np


class YOLOPose(YOLOPipeObjectBase):
    """
    YOLO姿态估计Pipe。
    
    基于Ultralytics YOLO模型进行人体姿态估计，检测人体关键点及骨架连接。
    
    Example:
        >>> from easymlops.yolo.pose import YOLOPose
        >>> pipe = YOLOPose(model_name="yolo11n-pose")
        >>> results = pipe.transform(df)
    """
    
    def __init__(self, model_name="yolo11n-pose", model_path=None, conf=0.25, iou=0.45,
                 imgsz=640, device="cpu", save=False, show=False, kpt_line=True, **kwargs):
        """
        初始化姿态估计Pipe。
        
        Args:
            model_name: 模型名称，如 yolo11n-pose, yolo11s-pose 等
            model_path: 模型文件路径，如果指定则优先使用
            conf: 置信度阈值
            iou: NMS IoU阈值
            imgsz: 输入图像尺寸
            device: 运行设备，auto/cpu/cuda
            save: 是否保存结果
            show: 是否显示结果
            kpt_line: 是否绘制骨架连接线
            **kwargs: 其他父类参数
        """
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            task_type="pose",
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save=save,
            show=show,
            **kwargs
        )
        self.kpt_line = kpt_line

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
        批量姿态估计。
        
        Args:
            s: 输入数据，DataFrame，包含image_path列
            
        Returns:
            包含姿态估计结果的DataFrame
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
            
            keypoints = result.keypoints
            boxes = result.boxes
            
            if keypoints is not None and len(keypoints) > 0:
                for i, kpt in enumerate(keypoints):
                    box = boxes[i] if boxes is not None and i < len(boxes) else None
                    
                    kpt_data = kpt.data.cpu().numpy()
                    
                    result_dict = {
                        "image_path": image_path,
                        "person_id": i,
                        "confidence": float(box.conf[0]) if box is not None else None,
                        "bbox_x1": float(box.xyxy[0][0]) if box is not None else None,
                        "bbox_y1": float(box.xyxy[0][1]) if box is not None else None,
                        "bbox_x2": float(box.xyxy[0][2]) if box is not None else None,
                        "bbox_y2": float(box.xyxy[0][3]) if box is not None else None,
                        "keypoints": kpt_data[0].tolist() if len(kpt_data) > 0 else None
                    }
                    results_list.append(result_dict)
            else:
                results_list.append({
                    "image_path": image_path,
                    "person_id": None,
                    "confidence": None,
                    "bbox_x1": None,
                    "bbox_y1": None,
                    "bbox_x2": None,
                    "bbox_y2": None,
                    "keypoints": None
                })
        
        return pd.DataFrame(results_list)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单张图像姿态估计。
        
        Args:
            s: 输入数据，字典，包含image_path键
            
        Returns:
            包含姿态估计结果的字典
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
        
        keypoints = result.keypoints
        boxes = result.boxes
        
        poses = []
        if keypoints is not None and len(keypoints) > 0:
            for i, kpt in enumerate(keypoints):
                box = boxes[i] if boxes is not None and i < len(boxes) else None
                
                kpt_data = kpt.data.cpu().numpy()
                
                poses.append({
                    "person_id": i,
                    "confidence": float(box.conf[0]) if box is not None else None,
                    "bbox": [float(x) for x in box.xyxy[0].tolist()] if box is not None else None,
                    "keypoints": kpt_data[0].tolist() if len(kpt_data) > 0 else None
                })
        
        s["poses"] = poses
        return s
