# -*- coding: utf-8 -*-
# RapidOCRText 验收: 真实图片 OCR/输出契约/save-load/单条推理/管道集成 (v0.2.8+)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd

from easymlops.ocr import RapidOCRText, OCRPipeLine

PASS, FAIL = [], []
IMG = "d:/datas/share/traes/easymlops-trae/tests/_ocr_sample.png"
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_rapidocr.pkl"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


def make_sample_image():
    """生成含清晰文字的测试图片"""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(img)
    text = "Hello easymlops 12345"
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 70), text, fill="black", font=font)
    img.save(IMG)
    return text


try:
    from rapidocr import RapidOCR  # noqa: F401
    HAS_RAPIDOCR = True
except ImportError:
    HAS_RAPIDOCR = False

if not HAS_RAPIDOCR:
    print("[SKIP] rapidocr not installed")
    sys.exit(0)

make_sample_image()
df = pd.DataFrame({"image_path": [IMG]})
required_cols = {"image_path", "text", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}

# ============ 1. 批量 OCR 契约 ============
print("=== 1. batch transform ===")
pipe = RapidOCRText(text_score=0.3)
pipe.fit(df)
out = pipe.transform(df.copy())
check("ocr_output_contract", required_cols.issubset(set(out.columns)) and len(out) >= 1)
joined = " ".join(out["text"].tolist()).lower()
check("ocr_recognizes_text", "easymlops" in joined and "12345" in joined,
      "recognized: {}".format(" | ".join(out["text"].tolist())[:80]))
check("ocr_confidence_range", ((out["confidence"] >= 0) & (out["confidence"] <= 1)).all())

# ============ 2. save/load ============
print("=== 2. save/load ===")
pipe.save(PKL)
pipe2 = RapidOCRText()
pipe2.load(PKL)
out2 = pipe2.transform(df.copy())
check("ocr_save_load_consistent",
      list(out2["text"]) == list(out["text"]) and
      np.allclose(out["confidence"].to_numpy(), out2["confidence"].to_numpy(), atol=1e-6))

# ============ 3. 单条推理 ============
print("=== 3. transform_single ===")
s = pipe.transform_single(dict(df.iloc[0]))
check("ocr_single", "text" in s and "all_texts" in s and
      "easymlops" in s["text"].lower())

# ============ 4. OCRPipeLine 集成 ============
print("=== 4. OCRPipeLine integration ===")
pipeline = OCRPipeLine()
pipeline.pipe(RapidOCRText(text_score=0.3))
pipeline.fit(df)
pout = pipeline.transform(df.copy())
check("ocr_pipeline_integration", required_cols.issubset(set(pout.columns)))

if os.path.exists(IMG):
    os.remove(IMG)
if os.path.exists(PKL):
    os.remove(PKL)
print("")
print("================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
