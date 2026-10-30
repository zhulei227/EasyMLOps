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


class RapidOCRText(OCRPipeObjectBase):
    """
    RapidOCR 文本检测与识别 Pipe（v0.2.8+ 新增）。

    基于 RapidOCR（PaddleOCR 模型的 ONNX 化轻量推理，支持
    ONNX Runtime/OpenVINO/TensorRT 等多后端），CPU 推理 20-50ms 级，
    无深度学习框架依赖——EasyOCR 的生产级替代（EasyOCR 推理慢且精度上限低）。

    输出契约与 EasyOCRText 对齐：image_path/text/confidence/bbox_x1..bbox_y2。

    Args:
        model_name: 模型名称
        model_path: 自定义模型目录（透传 RapidOCR 的 model_root_dir；None 用内置/默认下载）
        lang: 保留接口兼容（RapidOCR 由模型文件决定语言，中英文默认）
        text_score: 识别置信度阈值（默认 0.5，低于该值的文本框被过滤）
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.ocr import RapidOCRText
        >>> pipe = RapidOCRText()
        >>> df = pd.DataFrame({"image_path": ["receipt.png"]})
        >>> result = pipe.fit(df).transform(df)
    """

    def __init__(self, model_name="rapidocr", model_path=None, lang="ch",
                 text_score=0.5, **kwargs):
        kwargs.setdefault("gpu", False)   # RapidOCR 默认 ONNX CPU 后端
        super().__init__(model_name=model_name, model_path=model_path,
                         lang=lang, **kwargs)
        self.text_score = text_score

    def udf_get_params(self):
        """获取参数。"""
        params = super().udf_get_params()
        params.update({"text_score": self.text_score})
        return params

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        super().udf_set_params(params)
        self.text_score = params.get("text_score", self.text_score)
        # 引擎对象不可 pickle, load 后按恢复的参数重建
        self.udf_fit(None)

    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        加载模型（兼容 rapidocr 新包与 rapidocr_onnxruntime 旧包）。

        Args:
            s: 输入数据，DataFrame
            **kwargs: 其他参数

        Returns:
            self
        """
        params = {}
        if self.model_path:
            params["Global.model_root_dir"] = self.model_path
        try:
            from rapidocr import RapidOCR   # 新包(>=2.x/3.x)
            self.model = RapidOCR(params=params) if params else RapidOCR()
        except ImportError:
            try:
                from rapidocr_onnxruntime import RapidOCR   # 旧包
                self.model = RapidOCR()
            except ImportError:
                raise ImportError("请安装 rapidocr: pip install rapidocr onnxruntime")
        return self

    def _run_engine(self, image_path):
        """调用引擎并归一化输出为 [{box, text, score}]（兼容新旧 API）。"""
        raw = self.model(image_path)
        # 旧包返回 (result, elapse)
        if isinstance(raw, tuple) and len(raw) == 2 and (raw[0] is None or isinstance(raw[0], list)):
            raw = raw[0]
        if raw is None:
            return []
        if hasattr(raw, "txts"):   # 新包 RapidOCRResult
            boxes = raw.boxes if raw.boxes is not None else []
            txts = raw.txts if raw.txts is not None else []
            scores = raw.scores if raw.scores is not None else []
            return [{"box": box, "text": text, "score": score}
                    for box, text, score in zip(boxes, txts, scores)]
        items = []
        for item in raw:
            box, text, score = item[0], item[1], item[2]
            items.append({"box": box, "text": text, "score": score})
        return items

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量OCR识别。

        Args:
            s: 输入数据，DataFrame，包含 image_path 列

        Returns:
            包含OCR结果的DataFrame（image_path/text/confidence/bbox_x1..bbox_y2）
        """
        results_list = []
        for idx, row in s.iterrows():
            image_path = row.get("image_path", row.get("path", row.get("image", None)))
            if image_path is None:
                raise ValueError("未找到图像路径列，请确保数据包含image_path/path/image列")
            for item in self._run_engine(image_path):
                if float(item["score"]) < self.text_score:
                    continue
                box = np.asarray(item["box"], dtype=float)
                results_list.append({
                    "image_path": image_path,
                    "text": item["text"],
                    "confidence": float(item["score"]),
                    "bbox_x1": float(box[:, 0].min()),
                    "bbox_y1": float(box[:, 1].min()),
                    "bbox_x2": float(box[:, 0].max()),
                    "bbox_y2": float(box[:, 1].max()),
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

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        # OCR 语义为"一行输入 -> 多行输出", 与框架 extract_dict 逻辑不兼容, 直接走 udf
        return self.udf_transform_single(dict(s), **kwargs)
