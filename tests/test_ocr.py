import sys
import os
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd

test_images_dir = "d:/datas/share/traes/easymlops-trae/data/yolo_test_images"
bus_img = f"{test_images_dir}/bus.jpg"

print("测试 EasyOCRText...")
try:
    from easymlops.ocr import EasyOCRText
    
    df = pd.DataFrame({"image_path": [bus_img]})
    pipe = EasyOCRText(lang="en", gpu=False, batch_size=1)
    result = pipe.fit(df).transform(df)
    print(f"✅ 检测到 {len(result)} 个文本区域")
    print(result.head())
except ImportError as e:
    print(f"❌ 需要安装 easyocr: {e}")
except Exception as e:
    print(f"❌ 错误: {str(e)[:100]}")
