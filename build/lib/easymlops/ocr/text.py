from easymlops.ocr.core.pipe import OCRPipeObjectBase
from easymlops.table.core import dataframe_type, dict_type
import pandas as pd
import numpy as np


class EasyOCRText(OCRPipeObjectBase):
    """
    EasyOCR 文本检测与识别 Pipe。
    
    基于 EasyOCR 进行文本检测和识别，支持80+语言。
    
    Example:
        >>> from easymlops.ocr import EasyOCRText
        >>> pipe = EasyOCRText(lang=['en', 'ch_sim'])
        >>> df = pd.DataFrame({"image_path": ["test.jpg"]})
        >>> result = pipe.fit(df).transform(df)
    """
    
    def __init__(self, model_name="easyocr", model_path=None, lang="en",
                 gpu=True, batch_size=1, detail=1, 
                 text_threshold=0.7, link_threshold=0.4, 
                 canvas_size=2560, mag_ratio=1.0, **kwargs):
        """
        初始化 EasyOCR Pipe。
        
        Args:
            model_name: 模型名称
            model_path: 模型文件路径，如果指定则优先使用
            lang: 语言代码，支持单语言如 'en' 或多语言如 ['en', 'ch_sim', 'ja']
            gpu: 是否使用GPU加速
            batch_size: 批处理大小
            detail: 返回详细程度，0-简单结果，1-详细结果
            text_threshold: 文本检测阈值
            link_threshold: 文本链接阈值
            canvas_size: 图像处理最大尺寸
            mag_ratio: 图像放大比例
            **kwargs: 其他父类参数
        """
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            lang=lang,
            gpu=gpu,
            batch_size=batch_size,
            **kwargs
        )
        self.detail = detail
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio

    def udf_get_params(self):
        """获取参数。"""
        params = super().udf_get_params()
        params.update({
            "detail": self.detail,
            "text_threshold": self.text_threshold,
            "link_threshold": self.link_threshold,
            "canvas_size": self.canvas_size,
            "mag_ratio": self.mag_ratio
        })
        return params

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        super().udf_set_params(params)
        self.detail = params.get("detail", self.detail)
        self.text_threshold = params.get("text_threshold", self.text_threshold)
        self.link_threshold = params.get("link_threshold", self.link_threshold)
        self.canvas_size = params.get("canvas_size", self.canvas_size)
        self.mag_ratio = params.get("mag_ratio", self.mag_ratio)

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        加载模型。
        
        Args:
            s: 输入数据，DataFrame
            **kwargs: 其他参数
            
        Returns:
            self
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError("请安装easyocr: pip install easyocr")
        
        lang_list = self.lang if isinstance(self.lang, list) else [self.lang]
        
        self.model = easyocr.Reader(
            lang_list,
            gpu=self.gpu,
            model_storage_directory=self.model_path,
            download_enabled=self.model_path is None
        )
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量OCR识别。
        
        Args:
            s: 输入数据，DataFrame，包含image_path列
            
        Returns:
            包含OCR结果的DataFrame
        """
        results_list = []
        for idx, row in s.iterrows():
            image_path = row.get("image_path", row.get("path", row.get("image", None)))
            if image_path is None:
                raise ValueError("未找到图像路径列，请确保数据包含image_path/path/image列")
            
            result = self.model.readtext(
                image_path,
                batch_size=self.batch_size,
                detail=self.detail,
                text_threshold=self.text_threshold,
                link_threshold=self.link_threshold,
                canvas_size=self.canvas_size,
                mag_ratio=self.mag_ratio
            )
            
            if self.detail == 0:
                texts = result if isinstance(result, list) else []
                for text in texts:
                    results_list.append({
                        "image_path": image_path,
                        "text": text
                    })
            else:
                for item in result:
                    bbox = item[0]
                    text = item[1]
                    confidence = item[2]
                    
                    x1 = min([p[0] for p in bbox])
                    y1 = min([p[1] for p in bbox])
                    x2 = max([p[0] for p in bbox])
                    y2 = max([p[1] for p in bbox])
                    
                    results_list.append({
                        "image_path": image_path,
                        "text": text,
                        "confidence": float(confidence),
                        "bbox_x1": float(x1),
                        "bbox_y1": float(y1),
                        "bbox_x2": float(x2),
                        "bbox_y2": float(y2)
                    })
        
        if not results_list:
            results_list.append({"image_path": "", "text": "", "confidence": 0.0})
        
        return pd.DataFrame(results_list)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单张图像OCR识别。
        
        Args:
            s: 输入数据，dict，包含image_path键
            
        Returns:
            包含OCR结果的dict
        """
        df = pd.DataFrame([s])
        result_df = self.transform(df)
        
        if len(result_df) == 0 or (len(result_df) == 1 and result_df.iloc[0].get("text", "") == ""):
            return {"text": "", "all_texts": []}
        
        texts = result_df["text"].tolist()
        return {
            "text": " ".join(texts),
            "all_texts": texts,
            "detail": result_df.to_dict("records")
        }
