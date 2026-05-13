import numpy as np

# visualization.py 
def plot_geospatial_heatmap(X, y_true, y_pred, lat_idx, lon_idx, grid_size=10, save_path="geospatial_heatmap.png"):
    """
    draw the paragraph (10x10 grid) heatmap of prediction errors across the geographical area.
    parameter:
        X: Original unnormalized feature matrix( include latitude and longitude columns, assuming the columns are Longitude, Latitude)
        y_true: True house prices
        y_pred: Predicted house prices
        lat_idx: Latitude column index in X
        lon_idx: Longitude column index in X
        grid_size: Number of grid divisions (default is 10)
    """
    import matplotlib.pyplot as plt
    lons = X[:, lon_idx]
    lats = X[:, lat_idx]
    errors = np.abs(y_true - y_pred)
    
    # Define the latitude and longitude range and grid boundaries.
    lon_bins = np.linspace(lons.min(), lons.max(), grid_size + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), grid_size + 1)
    
    # create a grid to accumulate errors and counts
    grid_error = np.zeros((grid_size, grid_size))
    grid_counts = np.zeros((grid_size, grid_size))
    
    for i in range(len(lons)):
        lon_idx_grid = np.digitize(lons[i], lon_bins) - 1
        lat_idx_grid = np.digitize(lats[i], lat_bins) - 1
        if 0 <= lon_idx_grid < grid_size and 0 <= lat_idx_grid < grid_size:
            grid_error[lat_idx_grid, lon_idx_grid] += errors[i]
            grid_counts[lat_idx_grid, lon_idx_grid] += 1
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        grid_error = np.divide(grid_error, grid_counts, out=np.zeros_like(grid_error), where=grid_counts > 0)
    
    # draw the heatmap
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
    print(f"Geospatial heatmap saved as {save_path}")