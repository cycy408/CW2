"""
visualization.py - 可视化模块
包含：
- 绘制 MSE 对比柱状图
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mse_comparison(mse_dict, save_path="output/mse_comparison.png", title="Feature Selection Methods vs Test MSE"):
    """
    绘制不同模型的 MSE 柱状图
    mse_dict: 字典，键为模型名称（字符串），值为 MSE 数值
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    models = list(mse_dict.keys())
    mse_values = list(mse_dict.values())
    
    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)]
    bars = plt.bar(models, mse_values, color=colors)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=15, ha='right')
    
    # 在柱子上方标注数值
    for bar, mse in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{mse:.4f}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"柱状图已保存至 {save_path}")