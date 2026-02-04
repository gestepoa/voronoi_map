import io
import geopandas as gpd
import matplotlib
# 设置后端为 Agg，非交互式，适用于Web服务
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.spatial import cKDTree

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 

def geo_to_cartesian(lat_deg, lon_deg):
    """
    将经纬度转换为单位球体上的 3D 直角坐标 (x, y, z)
    """
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))

def generate_voronoi_map(
    points: list, 
    geojson_path: str, 
    extent: list,  # [min_lon, max_lon, min_lat, max_lat] or None
    point_colors: list,
    plugin_path: str = None,
    draw_plugin: bool = False,
    fill_ocean: bool = True,
    lat_n: int = 1200, 
    lon_n: int = 2400
):
    """
    生成 Voronoi 地图的核心函数 (包含局部网格优化 + 智能范围)
    """
    # 1. 读取主地图数据
    try:
        world = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"读取主地图失败: {e}")
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # --- 智能范围计算逻辑 ---
    # 如果前端传来的 extent 是空的 (None)，则根据 GeoJSON 自动计算边界
    if not extent or len(extent) != 4:
        print("未指定手动范围，正在根据 GeoJSON 自动计算边界...")
        min_lon, min_lat, max_lon, max_lat = world.total_bounds
        
        # 向外扩充 5% 的缓冲，避免贴边
        margin_x = (max_lon - min_lon) * 0.05
        margin_y = (max_lat - min_lat) * 0.05
        
        extent = [
            max(-180, min_lon - margin_x), 
            min(180, max_lon + margin_x), 
            max(-90, min_lat - margin_y), 
            min(90, max_lat + margin_y)
        ]
        print(f"自动应用范围: {extent}")

    # 2. 准备点位数据
    cities_coords = np.array(points)
    
    # 3. 智能网格生成 (仅生成视图范围内的网格)
    view_min_lon, view_max_lon, view_min_lat, view_max_lat = extent
    
    # 稍微多生成一点点网格(10%)，防止边缘出现空白缝隙
    grid_pad_x = (view_max_lon - view_min_lon) * 0.1
    grid_pad_y = (view_max_lat - view_min_lat) * 0.1
    
    print(f"正在生成局部优化网格 ({lat_n}x{lon_n})...")
    lats = np.linspace(max(-90, view_min_lat - grid_pad_y), min(90, view_max_lat + grid_pad_y), lat_n)
    lons = np.linspace(max(-180, view_min_lon - grid_pad_x), min(180, view_max_lon + grid_pad_x), lon_n)
    
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 坐标转换
    grid_points_cartesian = geo_to_cartesian(lat_grid.ravel(), lon_grid.ravel())
    city_points_cartesian = geo_to_cartesian(cities_coords[:, 0], cities_coords[:, 1])

    # 4. KDTree 计算最近邻
    tree = cKDTree(city_points_cartesian)
    dists, indices = tree.query(grid_points_cartesian, k=1)
    nearest_city = indices.reshape(lon_grid.shape)

    # 5. 绘图初始化
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': ccrs.Miller(central_longitude=150)})

    use_colors = point_colors[:len(points)]

    # (A) 绘制 Voronoi 填色区域
    ax.contourf(lon_grid, lat_grid, nearest_city, 
                levels=np.arange(-0.5, len(points), 1), 
                colors=use_colors, 
                transform=ccrs.PlateCarree(), 
                zorder=0)

    # (B) 绘制 Voronoi 边界线
    ax.contour(lon_grid, lat_grid, nearest_city, 
               levels=np.arange(0.5, len(points)-0.5, 1), 
               colors='k', 
               transform=ccrs.PlateCarree())

    # (C) 绘制 GeoJSON 主底图
    world.plot(ax=ax, color='none', edgecolor='black', linewidth=0.6, transform=ccrs.PlateCarree())

    # (D) 绘制辅助地图
    if draw_plugin and plugin_path:
        try:
            plugin_map = gpd.read_file(plugin_path)
            plugin_map.plot(ax=ax, color='none', edgecolor='dimgray', linewidth=0.2, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"辅助地图绘制失败: {e}")

    # (E) 绘制点位
    ax.scatter(cities_coords[:, 1], cities_coords[:, 0], 
               color="none", edgecolor="k", marker='o', s=50, zorder=5, 
               transform=ccrs.PlateCarree())

    # (F) 处理海洋填充
    if fill_ocean:
        ax.add_feature(cfeature.OCEAN, facecolor='k', zorder=0)

    # (G) 设置范围 (使用 extent，无论是手动的还是自动计算的)
    try:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    except Exception as e:
        print(f"设置范围失败: {e}")

    ax.tick_params(axis='both', which='both', length=0, labelsize=0)

    # 6. 保存结果
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf