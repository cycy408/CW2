import numpy as np

# visualization.py 新增函数
def plot_geospatial_heatmap(X, y_true, y_pred, lat_idx, lon_idx, grid_size=10, save_path="geospatial_heatmap.png"):
    """
    绘制预测误差的地理热图(10x10网格)
    参数：
        X: 原始未标准化的特征矩阵（必须包含经纬度列，假设列为 Longitude, Latitude）
        y_true: 真实房价
        y_pred: 预测房价
        lat_idx: 纬度在 X 中的列索引
        lon_idx: 经度在 X 中的列索引
        grid_size: 网格划分数量(默认10)
    """
    import matplotlib.pyplot as plt
    lons = X[:, lon_idx]
    lats = X[:, lat_idx]
    errors = np.abs(y_true - y_pred)
    
    # 划分网格边界
    lon_bins = np.linspace(lons.min(), lons.max(), grid_size + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), grid_size + 1)
    
    # 创建网格存储平均误差
    grid_error = np.zeros((grid_size, grid_size))
    grid_counts = np.zeros((grid_size, grid_size))
    
    for i in range(len(lons)):
        lon_idx_grid = np.digitize(lons[i], lon_bins) - 1
        lat_idx_grid = np.digitize(lats[i], lat_bins) - 1
        if 0 <= lon_idx_grid < grid_size and 0 <= lat_idx_grid < grid_size:
            grid_error[lat_idx_grid, lon_idx_grid] += errors[i]
            grid_counts[lat_idx_grid, lon_idx_grid] += 1
    
    # 避免除以零
    with np.errstate(divide='ignore', invalid='ignore'):
        grid_error = np.divide(grid_error, grid_counts, out=np.zeros_like(grid_error), where=grid_counts > 0)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(grid_error, origin='lower', cmap='hot', aspect='auto', 
                    extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]])
    plt.colorbar(im, label='Mean Absolute Prediction Error')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'{grid_size}x{grid_size} Geospatial Prediction Error Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"地理热图已保存为 {save_path}")