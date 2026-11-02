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
# ============ 5. PaddleOCRText(官方引擎对照) ============
print("=== 5. PaddleOCRText ===")
try:
    from easymlops.ocr import PaddleOCRText
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

if HAS_PADDLE:
    from PIL import Image, ImageDraw, ImageFont
    img_p = Image.new("RGB", (600, 200), "white")
    d = ImageDraw.Draw(img_p)
    try:
        pfont = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        pfont = ImageFont.load_default()
    d.text((30, 70), "Hello easymlops 12345", fill="black", font=pfont)
    PIMG = "d:/datas/share/traes/easymlops-trae/tests/_paddle_sample.png"
    img_p.save(PIMG)
    pdf = pd.DataFrame({"image_path": [PIMG]})

    ppipe = PaddleOCRText(lang="en", text_score=0.3)
    ppipe.fit(pdf)
    pout = ppipe.transform(pdf.copy())
    check("paddle_output_contract", required_cols.issubset(set(pout.columns)) and len(pout) >= 1)
    pjoined = " ".join(pout["text"].tolist()).lower()
    check("paddle_recognizes_text", "easymlops" in pjoined and "12345" in pjoined,
          "recognized: {}".format(" | ".join(pout["text"].tolist())[:80]))
    ppipe.save(PKL)
    ppipe2 = PaddleOCRText()
    ppipe2.load(PKL)
    pout2 = ppipe2.transform(pdf.copy())
    check("paddle_save_load", list(pout2["text"]) == list(pout["text"]))
    ps = ppipe.transform_single(dict(pdf.iloc[0]))
    check("paddle_single", "text" in ps and "easymlops" in ps["text"].lower())
    if os.path.exists(PIMG):
        os.remove(PIMG)
else:
    print("[SKIP] paddleocr not installed")

# ============ 6. PaddleOCRStructure(版面/表格解析) ============
print("=== 6. PaddleOCRStructure ===")
try:
    from easymlops.ocr import PaddleOCRStructure
    HAS_STRUCT = True
except ImportError:
    HAS_STRUCT = False

if HAS_STRUCT:
    # 复用 _paddle_sample.png(纯文本) + 生成带表格的图
    SIMG = "d:/datas/share/traes/easymlops-trae/tests/_structure_sample.png"
    from PIL import Image, ImageDraw, ImageFont
    img_s = Image.new("RGB", (500, 300), "white")
    ds = ImageDraw.Draw(img_s)
    ds.rectangle([20, 20, 280, 160], outline="black", width=2)
    ds.line([20, 90, 280, 90], fill="black", width=2)
    ds.line([150, 20, 150, 160], fill="black", width=2)
    try:
        sfont = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        sfont = ImageFont.load_default()
    ds.text((40, 45), "Name", fill="black", font=sfont)
    ds.text((170, 45), "Age", fill="black", font=sfont)
    ds.text((40, 115), "Tom", fill="black", font=sfont)
    ds.text((170, 115), "25", fill="black", font=sfont)
    ds.text((300, 40), "Hello document", fill="black", font=sfont)
    img_s.save(SIMG)
    sdf = pd.DataFrame({"image_path": [SIMG]})

    sengine = PaddleOCRStructure(lang="en")
    sengine.fit(sdf)
    sout = sengine.transform(sdf.copy())
    srequired = {"image_path", "type", "text", "html", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    check("structure_contract", srequired.issubset(set(sout.columns)) and len(sout) >= 1)
    tables = sout[sout["type"] == "table"]
    check("structure_table_detected", len(tables) >= 1 and
          any("<table" in str(h).lower() for h in tables["html"]),
          "regions: {}".format(sout["type"].tolist()))
    md = PaddleOCRStructure.to_markdown(sout)
    check("structure_to_markdown", isinstance(md, str) and len(md) > 0)
    sengine.save(PKL)
    sengine2 = PaddleOCRStructure()
    sengine2.load(PKL)
    sout2 = sengine2.transform(sdf.copy())
    check("structure_save_load", sout2["type"].tolist() == sout["type"].tolist())
    ss = sengine.transform_single(dict(sdf.iloc[0]))
    check("structure_single", "markdown" in ss and "regions" in ss)
    if os.path.exists(SIMG):
        os.remove(SIMG)
else:
    print("[SKIP] paddleocr not installed")

# ============ 7. VLDocumentParser(OpenAI 兼容适配器, 本地 mock 服务) ============
print("=== 7. VLDocumentParser ===")
try:
    from easymlops.ocr import VLDocumentParser
    import threading
    import json as _json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _MockVLMServer(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = _json.loads(self.rfile.read(length).decode("utf-8"))
            # 校验请求形态(OpenAI 兼容)
            assert body["model"] == "mock-vlm"
            assert body["messages"][0]["content"][0]["type"] == "text"
            assert body["messages"][0]["content"][1]["type"] == "image_url"
            content = "# Parsed Document\n\n| Name | Age |\n|------|-----|\n| Tom | 25 |"
            resp = {"choices": [{"message": {"role": "assistant", "content": content}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(_json.dumps(resp).encode("utf-8"))

        def log_message(self, *args):
            pass

    mock_server = HTTPServer(("127.0.0.1", 0), _MockVLMServer)
    mock_port = mock_server.server_address[1]
    threading.Thread(target=mock_server.serve_forever, daemon=True).start()
    HAS_STRUCT = True
except ImportError:
    HAS_STRUCT = False

if HAS_STRUCT:
    # 用回测试图片
    from PIL import Image, ImageDraw, ImageFont
    img_v = Image.new("RGB", (400, 120), "white")
    dv = ImageDraw.Draw(img_v)
    try:
        vfont = ImageFont.truetype("arial.ttf", 32)
    except Exception:
        vfont = ImageFont.load_default()
    dv.text((20, 40), "mock doc", fill="black", font=vfont)
    VIMG = "d:/datas/share/traes/easymlops-trae/tests/_vlm_sample.png"
    img_v.save(VIMG)
    vdf = pd.DataFrame({"image_path": [VIMG]})

    vpipe = VLDocumentParser(api_base="http://127.0.0.1:{}".format(mock_port),
                             model="mock-vlm", timeout=30)
    vpipe.fit(vdf)
    vout = vpipe.transform(vdf.copy())
    check("vlm_markdown_output", "markdown" in vout.columns and
          "Parsed Document" in vout["markdown"].iloc[0])
    vs = vpipe.transform_single(dict(vdf.iloc[0]))
    check("vlm_single", "markdown" in vs and "Parsed Document" in vs["markdown"])
    vpipe.save(PKL)
    vpipe2 = VLDocumentParser()
    vpipe2.load(PKL)
    check("vlm_save_load", vpipe2.model == "mock-vlm" and
          "Parsed Document" in vpipe2.transform(vdf.copy())["markdown"].iloc[0])
    mock_server.shutdown()
    if os.path.exists(VIMG):
        os.remove(VIMG)
else:
    print("[SKIP] easymlops.ocr.VLDocumentParser unavailable")

print("================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
