from easymlops.table.core import TablePipeLine
from easymlops.yolo.core.pipe import YOLOPipeObjectBase
import pandas as pd


class YOLOPipeline(TablePipeLine):
    """
    YOLO Pipeline。

    继承自TablePipeLine，专门用于YOLO相关任务的Pipeline。
    支持目标检测、实例分割、图像分类、姿态估计、旋转目标检测、训练、验证和导出。

    Example:
        >>> from easymlops import YOLOPipeline
        >>> from easymlops.yolo import YOLODetection
        >>>
        >>> pipeline = YOLOPipeline()
        >>> pipeline.pipe(YOLODetection(model_name="yolo11n"))
        >>> results = pipeline.fit(df).transform(df)
    """

    def __init__(self, pipes=None, *args, **kwargs):
        """
        初始化YOLO Pipeline。
        """
        super().__init__(*args, **kwargs)
        if pipes:
            for pipe in pipes:
                self.pipe(pipe)

    def pipe(self, pipe_obj: YOLOPipeObjectBase):
        """
        添加Pipe对象到Pipeline。

        Args:
            pipe_obj: YOLO Pipe对象

        Returns:
            self
        """
        return super().pipe(pipe_obj)

    def auto_test(self, x_, sample=100):
        """
        自动测试接口 - YOLO专用版本。

        由于YOLO transform可能输出一对多（一张图片多个检测结果），
        此方法跳过标准的一致性检查，仅验证功能可用性。

        :param x_: 输入数据
        :param sample: 采样数量
        :return:
        """
        import warnings
        warnings.filterwarnings("ignore")

        x = x_[:sample].sample(frac=1)

        print("=" * 60)
        print("YOLO Pipeline Auto Test")
        print("=" * 60)

        print("\n1. Fit test...")
        self.fit(x)
        print("   Fit passed")

        print("\n2. Transform test...")
        result = self.transform(x)
        print(f"   Transform passed: {len(result)} results")

        print("\n3. Get params test...")
        params = self.get_params()
        print("   Get params passed")

        print("\n4. Set params test...")
        self.set_params(params)
        print("   Set params passed")

        print("\n" + "=" * 60)
        print("All YOLO auto_test passed!")
        print("=" * 60)

    def load(self, path):
        """
        从path路径加载模型 - YOLO专用版本。

        由于YOLO模型（Ultralytics）无法直接序列化，
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
