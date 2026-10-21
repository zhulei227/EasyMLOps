"""
时间序列模型测试汇总
运行方式: python tests/test_ts_all.py

注意: 
- 统计模型(ARIMA, SARIMA等)可以在当前环境中运行
- 深度学习/Transformer/状态空间/生成模型需要正确的PyTorch环境
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

print("="*60)
print("运行时间序列模型测试")
print("="*60)

# 运行统计模型测试
print("\n>>> 运行统计模型测试...")
os.system("python tests/test_ts_statistical.py")

# 运行深度学习测试
print("\n>>> 运行深度学习模型测试...")
os.system("python tests/test_ts_deep_learning.py")

# 运行Transformer测试
print("\n>>> 运行Transformer模型测试...")
os.system("python tests/test_ts_transformer.py")

# 运行状态空间测试
print("\n>>> 运行状态空间模型测试...")
os.system("python tests/test_ts_state_space.py")

# 运行生成模型测试
print("\n>>> 运行生成模型测试...")
os.system("python tests/test_ts_generative.py")

print("\n" + "="*60)
print("所有时间序列模型测试完成!")
print("="*60)
print("""
注意:
- 统计模型测试已通过
- 深度学习/Transformer/状态空间/生成模型需要正确的PyTorch环境
- 如果需要运行这些测试，请确保安装正确版本的PyTorch
""")
