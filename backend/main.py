import os
import json
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# 引入静态文件处理
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from algorithm import generate_voronoi_map

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
UPLOAD_DIR = "../uploads"
FRONTEND_DIR = "../frontend"  # 前端代码所在目录
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 分辨率映射
RESOLUTION_MAP = {
    "low": (600, 1200),
    "medium": (1200, 2400),
    "high": (2400, 4800)
}

# --- 1. 核心 API 接口 ---
@app.post("/generate-map")
async def api_generate_map(
    points: str = Form(...),
    extent: str = Form(...),
    colors: str = Form(...),
    draw_plugin: bool = Form(False),
    fill_ocean: bool = Form(True),
    resolution: str = Form("medium"),
    file: UploadFile = File(...),
    plugin_file: UploadFile = File(None)
):
    try:
        points_list = json.loads(points)
        extent_list = json.loads(extent) # 可能为 None
        colors_list = json.loads(colors)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON format error")

    if len(points_list) < 2 or len(points_list) > 30:
        raise HTTPException(status_code=400, detail="Points count must be between 2 and 30")
    
    if len(colors_list) != len(points_list):
         raise HTTPException(status_code=400, detail="Colors count mismatch")

    # 保存主文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 保存辅助文件
    plugin_path = None
    if plugin_file:
        plugin_path = os.path.join(UPLOAD_DIR, "plugin_" + plugin_file.filename)
        with open(plugin_path, "wb") as buffer:
            shutil.copyfileobj(plugin_file.file, buffer)

    lat_n, lon_n = RESOLUTION_MAP.get(resolution, (1200, 2400))

    try:
        img_buf = await run_in_threadpool(
            generate_voronoi_map, 
            points_list, 
            file_path, 
            extent_list,
            colors_list,
            plugin_path,
            draw_plugin,
            fill_ocean,
            lat_n,
            lon_n
        )
        return StreamingResponse(img_buf, media_type="image/png")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. 静态资源托管 (解决局域网访问问题) ---

# 挂载 frontend 目录，使得 index.html 可以引用同目录下的其他文件(如果有)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# 访问根路径 http://ip:8000/ 直接显示首页
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" 监听所有网卡，port=8000
    uvicorn.run(app, host="0.0.0.0", port=8000)