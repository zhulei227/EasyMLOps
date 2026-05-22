import easyocr
# 创建OCR对象
reader = easyocr.Reader(['en'])
# 识别文字
result = reader.readtext('image.jpg')
# 处理识别结果
for (text, bbox, confidence) in result:
    print(f'Text: {text}, Bbox: {bbox}, Confidence: {confidence}')
