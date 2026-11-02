from easymlops.ocr.core.pipe import OCRPipeObjectBase
from easymlops.table.core import dataframe_type, dict_type
from easymlops.core.storage import is_remote_path, load_model_bytes
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

class PaddleOCRText(OCRPipeObjectBase):
    """
    PaddleOCR 文本检测与识别 Pipe（v0.2.9+ 新增）。

    基于 PaddleOCR 官方引擎（PP-OCR 系列，SOTA 精度、支持 111 种语言）。
    兼容 paddleocr 2.x(ocr.ocr API) 与 3.x(predict API) 两代接口；
    输出契约与 RapidOCRText/EasyOCRText 对齐：
    image_path/text/confidence/bbox_x1..bbox_y2。

    注意:
        - paddleocr 3.x 需 python>=3.9；python 3.8 请用 paddleocr==2.10.x + paddlepaddle>=3.0
        - 首次运行会自动下载 PP-OCR 模型（约 10-20MB）
        - 引擎不可 pickle，load 后按恢复的参数自动重建

    Args:
        model_name: 模型名称
        model_path: 自定义模型目录（透传 det/rec 模型路径；None 用默认下载）
        lang: 语言代码（'ch' 中英混合 / 'en' 英文 / 'japan' / 'korean' 等）
        use_angle_cls: 是否使用方向分类器矫正文本行
        text_score: 识别置信度阈值（默认 0.5）
        gpu: 是否使用 GPU
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.ocr import PaddleOCRText
        >>> pipe = PaddleOCRText(lang="ch", text_score=0.5)
        >>> df = pd.DataFrame({"image_path": ["receipt.png"]})
        >>> result = pipe.fit(df).transform(df)
    """

    def __init__(self, model_name="paddleocr", model_path=None, lang="ch",
                 use_angle_cls=True, text_score=0.5, **kwargs):
        super().__init__(model_name=model_name, model_path=model_path,
                         lang=lang, **kwargs)
        self.use_angle_cls = use_angle_cls
        self.text_score = text_score

    def udf_get_params(self):
        """获取参数。"""
        params = super().udf_get_params()
        params.update({"use_angle_cls": self.use_angle_cls, "text_score": self.text_score})
        return params

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        super().udf_set_params(params)
        self.use_angle_cls = params.get("use_angle_cls", self.use_angle_cls)
        self.text_score = params.get("text_score", self.text_score)
        # 引擎不可 pickle, load 后按恢复的参数重建
        self.udf_fit(None)


    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        加载模型（兼容 paddleocr 2.x 与 3.x 两代 API）。

        Args:
            s: 输入数据，DataFrame
            **kwargs: 其他参数

        Returns:
            self
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("请安装 paddleocr: pip install paddleocr paddlepaddle")
        import paddleocr
        kwargs_engine = {"lang": self.lang if isinstance(self.lang, str) else self.lang[0]}
        if self.model_path:
            kwargs_engine["det_model_dir"] = self.model_path
        if int(paddleocr.__version__.split(".")[0]) >= 3:
            # 3.x: predict API
            kwargs_engine.update({
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": self.use_angle_cls,
            })
        else:
            # 2.x: ocr.ocr API
            kwargs_engine.update({
                "use_angle_cls": self.use_angle_cls,
                "show_log": False,
            })
        self.model = PaddleOCR(**kwargs_engine)
        self._engine_major_version = int(paddleocr.__version__.split(".")[0])
        return self

    def _run_engine(self, image_path):
        """调用引擎并归一化输出为 [{box, text, score}]（兼容 2.x/3.x API）。"""
        items = []
        if getattr(self, "_engine_major_version", 2) >= 3:
            for res in self.model.predict(image_path):
                texts = res.get("rec_texts") or []
                scores = res.get("rec_scores") or []
                polys = res.get("rec_polys") or res.get("rec_boxes") or []
                for box, text, score in zip(polys, texts, scores):
                    items.append({"box": box, "text": text, "score": float(score)})
            return items
        raw = self.model.ocr(image_path, cls=self.use_angle_cls)
        # 2.x: 全局 list 或按图嵌套 list, 元素为 [box, (text, score)]
        if raw and isinstance(raw[0], list) and raw[0] and \
                isinstance(raw[0][0], list) and len(raw[0][0]) == 2:
            img_items = raw[0]
        else:
            img_items = raw or []
        for item in img_items:
            if item is None:
                continue
            box, (text, score) = item[0], item[1]
            items.append({"box": box, "text": text, "score": float(score)})
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
                if item["score"] < self.text_score:
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

class PaddleOCRStructure(OCRPipeObjectBase):
    """
    PaddleOCR 版面分析与表格解析 Pipe（v0.2.9+ 新增，PP-Structure）。

    对文档图片做版面分区（text/table/figure/title...），文本区域给出识别文本，
    表格区域给出 HTML 结构——面向 RAG/LLM 的文档结构化入口。

    输出字段: image_path / type / text / html / bbox_x1..bbox_y2
    另提供 `to_markdown(result_df)` 把分区结果转为阅读顺序的 Markdown 文档。

    注意:
        - 当前实现基于 paddleocr 2.x 的 PPStructure API（py3.8 环境）
        - 首次运行会自动下载版面/表格模型（约 20-30MB）
        - 引擎不可 pickle，load 后按恢复的参数自动重建

    Args:
        model_name: 模型名称
        lang: 语言代码（'ch' / 'en'）
        layout: 是否启用版面分析（False 时退化为纯 OCR 分区）
        table: 是否启用表格识别
        text_score: 识别置信度阈值（默认 0.5）
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.ocr import PaddleOCRStructure
        >>> pipe = PaddleOCRStructure(lang="ch")
        >>> df = pd.DataFrame({"image_path": ["doc.png"]})
        >>> result = pipe.fit(df).transform(df)
        >>> md = PaddleOCRStructure.to_markdown(result)
    """

    def __init__(self, model_name="paddle_structure", model_path=None, lang="ch",
                 layout=True, table=True, text_score=0.5, **kwargs):
        super().__init__(model_name=model_name, model_path=model_path,
                         lang=lang, **kwargs)
        self.layout = layout
        self.table = table
        self.text_score = text_score

    def udf_get_params(self):
        """获取参数。"""
        params = super().udf_get_params()
        params.update({"layout": self.layout, "table": self.table,
                       "text_score": self.text_score})
        return params

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        super().udf_set_params(params)
        self.layout = params.get("layout", self.layout)
        self.table = params.get("table", self.table)
        self.text_score = params.get("text_score", self.text_score)
        # 引擎不可 pickle, load 后按恢复的参数重建
        self.udf_fit(None)


    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        加载 PP-Structure 引擎（当前实现基于 paddleocr 2.x API）。

        Args:
            s: 输入数据，DataFrame
            **kwargs: 其他参数

        Returns:
            self
        """
        try:
            from paddleocr import PPStructure
        except ImportError:
            raise ImportError("请安装 paddleocr: pip install paddleocr paddlepaddle")
        import paddleocr
        if int(paddleocr.__version__.split(".")[0]) >= 3:
            raise ImportError("PaddleOCRStructure 当前支持 paddleocr 2.x; "
                              "3.x 请使用 PPStructureV3（python>=3.9）")
        self.model = PPStructure(layout=self.layout, table=self.table, ocr=True,
                                 lang=self.lang if isinstance(self.lang, str) else self.lang[0],
                                 show_log=False)
        return self

    def _run_engine(self, image_path):
        """调用引擎并归一化输出为 [{type, text, html, box}]。"""
        regions = self.model(image_path)
        items = []
        for region in regions or []:
            res = region.get("res") or {}
            text = ""
            if res.get("text"):
                text = res["text"]
            elif res.get("rec_texts"):
                text = " ".join(res["rec_texts"])
            html = res.get("html") or ""
            items.append({"type": region.get("type", "unknown"),
                          "text": text, "html": html,
                          "box": region.get("bbox", [0, 0, 0, 0])})
        return items

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量版面解析。

        Args:
            s: 输入数据，DataFrame，包含 image_path 列

        Returns:
            DataFrame: image_path/type/text/html/bbox_x1..bbox_y2
        """
        results_list = []
        for idx, row in s.iterrows():
            image_path = row.get("image_path", row.get("path", row.get("image", None)))
            if image_path is None:
                raise ValueError("未找到图像路径列，请确保数据包含image_path/path/image列")
            for item in self._run_engine(image_path):
                if item["type"] != "table" and item["text"] and self.text_score:
                    # 文本区域不做逐行分数过滤(引擎内部已过滤), 表格保留 html
                    pass
                box = np.asarray(item["box"], dtype=float).reshape(-1, 2) \
                    if np.asarray(item["box"]).ndim > 1 else np.asarray(item["box"], dtype=float)
                results_list.append({
                    "image_path": image_path,
                    "type": item["type"],
                    "text": item["text"],
                    "html": item["html"],
                    "bbox_x1": float(np.min(box)),
                    "bbox_y1": float(np.min(box)),
                    "bbox_x2": float(np.max(box)),
                    "bbox_y2": float(np.max(box)),
                })

        if not results_list:
            results_list.append({"image_path": "", "type": "", "text": "", "html": "",
                                 "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 0.0, "bbox_y2": 0.0})

        return pd.DataFrame(results_list)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单张图像版面解析。

        Args:
            s: 输入数据，dict，包含image_path键

        Returns:
            包含解析结果的dict
        """
        df = pd.DataFrame([s])
        result_df = self.transform(df)

        if len(result_df) == 0 or result_df.iloc[0].get("type", "") == "":
            return {"text": "", "markdown": "", "regions": []}

        regions = result_df.to_dict("records")
        return {
            "text": " ".join(r["text"] for r in regions if r["text"]),
            "markdown": self.to_markdown(result_df),
            "regions": regions,
        }

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        # OCR 语义为"一行输入 -> 多行输出", 与框架 extract_dict 逻辑不兼容, 直接走 udf
        return self.udf_transform_single(dict(s), **kwargs)

    @staticmethod
    def to_markdown(result_df: dataframe_type) -> str:
        """
        把版面解析结果转为阅读顺序（按 bbox 从上到下）的 Markdown 文档:
        文本区域输出识别文本, 表格区域输出 HTML 块。

        Args:
            result_df: transform 的输出 DataFrame

        Returns:
            str: Markdown 文本
        """
        df = result_df[result_df["type"] != ""].sort_values(
            ["bbox_y1", "bbox_x1"]).reset_index(drop=True)
        parts = []
        for _, row in df.iterrows():
            if row["type"] == "table" and row["html"]:
                parts.append(str(row["html"]))
            elif row["text"]:
                parts.append(str(row["text"]))
        return "\n\n".join(parts)

class VLDocumentParser(OCRPipeObjectBase):
    """
    VLM 文档解析适配器（v0.2.9+ 新增）：调用 **OpenAI 兼容 API** 的视觉语言模型
    做端到端 OCR/文档解析，输出 Markdown——对接 vLLM/LM Studio 等本地服务
    （如 DeepSeek-OCR、Qwen2.5-VL、GLM-OCR）或云端多模态 API。

    与传统引擎(RapidOCR/PaddleOCR)的分工:
        - VLM: 公式/表格/阅读顺序/手写等复杂版面的结构化输出(Markdown)
        - 传统引擎: 忠实度更高(调研实证: VLM 存在幻觉风险, CER 低但可能改写实体),
          忠实度攸关场景建议两者交叉校验

    Args:
        api_base: OpenAI 兼容服务地址（如 "http://localhost:8000/v1"）
        api_key: API Key（本地服务默认 "EMPTY"）
        model: 模型名（如 "deepseek-ocr"、"qwen2.5-vl-7b-instruct"）
        prompt: 解析指令（默认通用文档转 Markdown 指令）
        image_col: 输入图像路径列名（默认 "image_path"）
        temperature: 采样温度（OCR 场景建议 0）
        max_tokens: 最大生成 token 数
        timeout: 请求超时秒数
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.ocr import VLDocumentParser
        >>> pipe = VLDocumentParser(api_base="http://localhost:8000/v1",
        ...                         model="deepseek-ocr")
        >>> df = pd.DataFrame({"image_path": ["doc.png"]})
        >>> result = pipe.fit(df).transform(df)   # 追加 markdown 列
    """

    DEFAULT_PROMPT = ("You are an OCR engine. Convert ALL text in the image into "
                      "markdown, preserving reading order, tables as HTML tables, "
                      "and formulas as LaTeX. Output ONLY the markdown content, "
                      "no explanations.")

    def __init__(self, api_base="http://localhost:8000/v1", api_key="EMPTY", model=None,
                 prompt=None, image_col="image_path", temperature=0.0, max_tokens=4096,
                 timeout=120, **kwargs):
        super().__init__(**kwargs)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.image_col = image_col
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)

    def udf_get_params(self):
        """获取参数（无模型对象, 配置即全部状态）。"""
        params = super().udf_get_params()
        params.update({"api_base": self.api_base, "api_key": self.api_key,
                       "model": self.model, "prompt": self.prompt,
                       "image_col": self.image_col, "temperature": self.temperature,
                       "max_tokens": self.max_tokens, "timeout": self.timeout})
        return params

    def udf_set_params(self, params: dict_type):
        """设置参数。"""
        super().udf_set_params(params)
        self.api_base = params.get("api_base", self.api_base).rstrip("/")
        self.api_key = params.get("api_key", self.api_key)
        self.model = params.get("model", self.model)
        self.prompt = params.get("prompt", self.prompt)
        self.image_col = params.get("image_col", self.image_col)
        self.temperature = params.get("temperature", self.temperature)
        self.max_tokens = params.get("max_tokens", self.max_tokens)
        self.timeout = params.get("timeout", self.timeout)


    def udf_fit(self, s: dataframe_type, **kwargs):
        """
        校验配置（VLM 为外部服务, 无本地模型对象; 此处仅做参数完整性检查）。

        Args:
            s: 输入数据，DataFrame
            **kwargs: 其他参数

        Returns:
            self
        """
        if not self.api_base:
            raise Exception("VLDocumentParser requires api_base")
        if not self.model:
            raise Exception("VLDocumentParser requires model (e.g. 'deepseek-ocr')")
        return self

    def _image_to_data_uri(self, image_path: str) -> str:
        """把本地图片编码为 base64 data URI。"""
        import base64
        if is_remote_path(image_path):
            data = load_model_bytes(image_path)
        else:
            with open(image_path, "rb") as f:
                data = f.read()
        ext = str(image_path).split(".")[-1].lower()
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        return "data:image/{};base64,{}".format(mime, base64.b64encode(data).decode("ascii"))

    def _call_vlm(self, image_path: str) -> str:
        """调用 OpenAI 兼容 /chat/completions, 返回模型输出的 Markdown。"""
        import requests
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url",
                     "image_url": {"url": self._image_to_data_uri(image_path)}},
                ],
            }],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": "Bearer {}".format(self.api_key)}
        resp = requests.post(self.api_base + "/chat/completions", json=payload,
                             headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            raise Exception("VLM api error {}: {}".format(resp.status_code, resp.text[:200]))
        content = resp.json()["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else str(content)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        """
        批量 VLM 文档解析：对每行图片调用视觉模型，输出 Markdown。

        Args:
            s: 输入数据，DataFrame，包含图像路径列

        Returns:
            DataFrame: 原输入列 + markdown 列（模型输出的 Markdown 文本）
        """
        result = s.copy()
        markdowns = []
        for idx, row in s.iterrows():
            image_path = row.get(self.image_col, row.get("image_path",
                                                         row.get("path", None)))
            if image_path is None:
                raise ValueError("未找到图像路径列，请确保数据包含{}列".format(self.image_col))
            markdowns.append(self._call_vlm(image_path))
        result["markdown"] = markdowns
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        """
        单张图像 VLM 文档解析。

        Args:
            s: 输入数据，dict，包含图像路径键

        Returns:
            dict: 原输入键 + markdown 键
        """
        result = dict(s)
        image_path = s.get(self.image_col, s.get("image_path", s.get("path", None)))
        if image_path is None:
            raise ValueError("未找到图像路径键，请确保数据包含{}键".format(self.image_col))
        result["markdown"] = self._call_vlm(image_path)
        return result
