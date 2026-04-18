from easymlops.table.core import TablePipeLine
from typing import Union, List
import pandas as pd


class OCRPipeLine(TablePipeLine):
    """
    OCR Pipeline。

    继承自TablePipeLine，专门用于处理OCR任务流水线。

    Example:
        >>> from easymlops.ocr import OCRPipeLine
        >>> from easymlops.ocr import EasyOCRText
        >>> pipeline = OCRPipeLine()
        >>> pipeline.pipe(EasyOCRText(lang='en', gpu=False))
        >>> result = pipeline.fit(df).transform(df)
    """

    def __init__(self, pipes: List = None, n_jobs: int = 1, verbose: int = 1, **kwargs):
        """
        初始化OCR Pipeline。

        Args:
            pipes: Pipe对象列表
            n_jobs: 并行任务数，1为串行
            verbose: 日志详细程度
            **kwargs: 其他父类参数
        """
        super().__init__(**kwargs)
        self.pipes = pipes or []
        self.n_jobs = n_jobs
        self.verbose = verbose
        for pipe in self.pipes:
            self.add_pipe(pipe)

    def load(self, path):
        """
        从path路径加载模型 - OCR专用版本。

        由于OCR模型（EasyOCR）无法直接序列化，
        加载后需要用真实数据重新fit以恢复模型。

        :param path: 模型路径
        :return: self
        """
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            structs = data["structs"]
            params = data["params"]
            self.models = []
            for struct in structs:
                self.pipe(self.build_structs(struct))
            for i, param in enumerate(params):
                self.models[i].set_params(param)

        return self

    def auto_test(self, x_, sample=100):
        """
        自动测试接口 - OCR专用版本。

        由于OCR transform可能输出一对多（一张图片多个文本区域），
        此方法跳过标准的一致性检查，仅验证功能可用性。

        :param x_: 输入数据
        :param sample: 采样数量
        :return:
        """
        import warnings
        warnings.filterwarnings("ignore")

        x = x_[:sample].sample(frac=1)

        print("=" * 60)
        print("OCR Pipeline Auto Test")
        print("=" * 60)

        print("\n1. Fit test...")
        self.fit(x)
        print("   ✅ Fit passed")

        print("\n2. Transform test...")
        result = self.transform(x)
        print(f"   ✅ Transform passed: {len(result)} results")

        print("\n3. Get params test...")
        params = self.get_params()
        print(f"   ✅ Get params passed")

        print("\n4. Set params test...")
        self.set_params(params)
        print(f"   ✅ Set params passed")

        print("\n" + "=" * 60)
        print("All OCR auto_test passed!")
        print("=" * 60)
