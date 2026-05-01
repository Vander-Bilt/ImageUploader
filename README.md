# 📤 ComfyUI Image Uploader Node

> 将 ComfyUI 生成的图片批量上传到任意外部 API 的自定义节点

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📋 目录

- [功能特性](#-功能特性)
- [安装步骤](#-安装步骤)
- [快速开始](#-快速开始)
- [节点参数说明](#-节点参数说明)
- [返回值说明](#-返回值说明)
- [后端 API 要求](#-后端-api-要求)
- [工作流示例](#-工作流示例)
- [故障排除](#-故障排除)
- [开发指南](#-开发指南)
- [贡献](#-贡献)
- [许可证](#-许可证)

---

## ✨ 功能特性

- 🔄 **批量上传**：支持一次性上传多张图片（Batch 处理）
- 🔗 **灵活配置**：自定义 API 基础地址和上传端点
- 🧠 **特征提取**：支持上传图片特征向量（兼容 `CLIP_VISION_OUTPUT`），支持自动池化与半精度压缩（float16）
- 🏷️ **标签支持**：支持为图片添加多标签，提供灵活的解析规则（按行匹配或全量应用）
- 🖼️ **格式兼容**：自动将 ComfyUI 的 `[B,H,W,C]` Tensor 转换为 PNG 字节流
- 📦 **混合上传**：采用 `multipart/form-data` 格式，同时发送图片二进制、特征向量及元数据
- 🛡️ **错误处理**：单个图片上传失败不影响其他图片，错误信息清晰可见
- ⏱️ **超时配置**：默认 300 秒超时，支持大文件上传

---

## 📦 安装步骤

### 方式一：手动安装

```bash
# 1. 进入 ComfyUI 自定义节点目录
cd ComfyUI/custom_nodes/

# 2. 克隆本仓库（或下载单个文件）
git clone https://github.com/yourusername/comfyui-image-uploader.git

# 3. 安装依赖（如需）
cd comfyui-image-uploader
pip install -r requirements.txt  # 如果有的话

# 4. 重启 ComfyUI
```

### 方式二：ComfyUI Manager（推荐）

> 如果本节点已注册到 [ComfyUI Registry](https://registry.comfy.org/)，可直接在 Manager 中搜索 `ImageUploader` 安装。

---

## 🚀 快速开始

1. 在 ComfyUI 画布中右键 → `image/upload` → 选择 **📤 Upload Image to API**
2. 将 `VAE Decode` 或任何输出 `IMAGE` 的节点连接到本节点的 `images` 输入
3. 配置 API 地址参数：
   - `api_base_url`: `http://localhost:3011`
   - `upload_endpoint`: `/upload-image-binary`
4. 点击 **Queue Prompt** 执行工作流
5. 查看节点输出面板中的上传结果

---

## ⚙️ 节点参数说明

### 必需参数（Required）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `images` | `IMAGE` | - | 接收 ComfyUI 标准图片格式 `[B,H,W,C]` |
| `api_base_url` | `STRING` | `http://localhost:3011` | API 服务的基础地址 |
| `upload_endpoint` | `STRING` | `/upload-image-binary` | 上传接口的路径 |

### 可选参数（Optional）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `features` | `CLIP_VISION_OUTPUT` | - | 图片特征向量。支持自动池化（CLS/Mean）及 float16 压缩 |
| `labels` | `STRING` | - | 图片标签。详见 [标签解析规则](#-标签解析规则) |
| `batchInfo` | `STRING` | `{}` | 总体信息字段。JSON 字符串，包含该 batch 的共有属性（如分类、风格等） |

### 🚀 路线图（规划中）

- [ ] `timeout`: 支持自定义请求超时时间
- [ ] `headers`: 支持自定义 HTTP 请求头（用于 Token 认证等）
- [ ] `image_format`: 支持上传为 JPEG/WebP 格式以节省空间
- [ ] `retry_count`: 失败自动重试机制

---

## 🏷️ 标签解析规则

`labels` 输入框支持多种解析方式：

1. **全局标签**：输入一行文字（如 `cat, outdoor`），该 batch 中**所有图片**都将带上这些标签。
2. **逐行匹配**：输入多行文字，行数需与图片 Batch 数量一致。每行对应一张图片的标签。
3. **显式分隔**：使用 `|` 符号分隔每张图的标签组（如 `tag1,tag2 | tag3,tag4`）。

---

## 📤 返回值说明

节点输出两个端口，便于不同场景使用：

### 1. `urls`（STRING）
```
https://cdn.example.com/img1.png
https://cdn.example.com/img2.png
ERROR: Connection timeout
```
- 每行一个结果，成功为 URL，失败为 `ERROR: xxx`
- 适合连接 `Show Text` 节点预览或日志记录

### 2. `response_data`（JSON）
```json
{
  "total": 2,
  "success_count": 1,
  "results": [
    {
      "success": true,
      "data": { "url": "https://...", "id": "xxx" },
      "status_code": 200
    },
    {
      "success": false,
      "error": "Connection timeout",
      "data": null
    }
  ],
  "urls": [
    "https://cdn.example.com/img1.png",
    "ERROR: Connection timeout"
  ]
}
```
- 结构化数据，适合连接后续逻辑节点进行条件判断或数据处理

---

## 🔌 后端 API 要求

本节点采用 `multipart/form-data` 格式上传图片及关联元数据：

### 请求规范
- **Method**: `POST`
- **Body**:
  - `image`: 文件字段（PNG 格式）
  - `metadata`: 字符串字段。内容为 JSON 格式的元数据，包含：
    - `feature_vector`: 浮点数列表（如为 `None` 则不包含）
    - `labels`: 字符串列表（如为 `None` 则不包含）
    - `index`: 整数（图片在 Batch 中的索引）

### 响应规范（成功）
需包含 `url` 字段供节点解析：
```json
{
  "url": "https://cdn.example.com/uploaded-image.png",
  "id": "abc123"
}
```

### 示例：FastAPI 后端实现
```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json, uuid, os

app = FastAPI()

@app.post("/upload-image-binary")
async def upload_image(
    image: UploadFile = File(...),
    metadata: str = Form(...)
):
    # 1. 解析元数据
    meta_data = json.loads(metadata)
    feature_vector = meta_data.get("feature_vector")
    labels = meta_data.get("labels")
    
    # 2. 读取图片二进制流
    content = await image.read()
    
    # 3. 执行存储逻辑（示例）
    filename = f"{uuid.uuid4()}.png"
    with open(f"uploads/{filename}", "wb") as f:
        f.write(content)
    
    # 4. 返回结果
    return {
        "url": f"http://your-domain.com/uploads/{filename}",
        "metadata_received": {
            "label_count": len(labels) if labels else 0,
            "has_features": feature_vector is not None
        }
    }
```

---

## 🔄 工作流示例

```
[Load Checkpoint] 
       ↓
[CLIP Text Encode (Prompt)] 
       ↓
[KSampler] → [VAE Decode] → [📤 Upload Image to API]
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
           [Show Text: urls]              [JSON Parser: response_data]
                    ↓                                   ↓
           预览上传结果链接              提取 URL/ID 用于后续逻辑
```

### 高级用法：条件分支处理
```python
# 使用 "JSON Parser" 节点解析 response_data
# 连接 "Conditional Switch" 节点：
#   - success_count == total → 继续生成流程
#   - 否则 → 触发告警/重试逻辑
```

---

## 🛠️ 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `Connection refused` | API 服务未启动 / 地址错误 | 检查 `api_base_url` 是否正确，确认后端服务运行中 |
| `404 Not Found` | 端点路径错误 | 检查 `upload_endpoint` 是否以 `/` 开头，路径是否拼写正确 |
| `Upload failed: 413` | 图片过大 | 后端配置 `client_max_body_size`，或压缩图片（后续版本支持） |
| `JSON decode error` | 后端返回非 JSON | 确保成功/失败响应均为合法 JSON 格式 |
| 节点不显示输出 | 未设置 `OUTPUT_NODE = True` | 已内置，如自定义修改请保留该属性 |
| 内存占用高 | 批量图片过多 | 建议单次 Batch ≤ 10，或启用分片上传（规划中） |

### 调试技巧
```python
# 在 upload_images 函数开头添加：
import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug(f"Images shape: {images.shape}")
```

---

## 💻 开发指南

### 项目结构
```
comfyui-image-uploader/
├── __init__.py          # 节点注册入口
├── image_uploader.py    # 主逻辑（本文件）
├── requirements.txt     # Python 依赖（可选）
├── README.md            # 本文档
└── examples/
    ├── workflow.json    # 示例工作流
    └── mock_server.py   # 本地测试用 Mock 服务
```

### 本地测试 Mock 服务
```bash
# 启动测试服务器（5000 端口）
python examples/mock_server.py

# 配置节点参数：
#   api_base_url: http://localhost:5000
#   upload_endpoint: /upload
```

`mock_server.py` 示例：
```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    # 模拟上传延迟
    import time; time.sleep(1)
    return jsonify({
        "url": f"http://localhost:5000/files/test_{int(time.time())}.png",
        "mock": True
    })

if __name__ == '__main__':
    app.run(port=5000)
```

### 扩展开发
如需添加新功能（如认证、压缩、CDN 适配），建议：
1. 在 `optional` 参数中声明新配置项
2. 在 `upload_single_image` 中扩展逻辑
3. 保持向后兼容，默认值不影响现有用户

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！贡献前请：

1. ✅ 确保代码通过 `flake8` 基础检查
2. ✅ 在 `examples/` 中添加工作流示例（如适用）
3. ✅ 更新本文档的对应章节
4. ✅ 描述变更内容（Breaking Change 需特别说明）

```bash
# 开发环境设置
git clone https://github.com/yourusername/comfyui-image-uploader.git
cd comfyui-image-uploader
# 链接到 ComfyUI
ln -s $(pwd) /path/to/ComfyUI/custom_nodes/image-uploader
```

---

## 📜 许可证

本项目采用 [MIT License](LICENSE)。  
可自由使用、修改、商用，但请保留原始版权声明。

---

## 🙏 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的节点式工作流引擎
- [Pillow](https://python-pillow.org/) - 图像处理支持
- 所有贡献者和早期测试用户 🎉

---

> 💡 **小提示**：上传敏感图片时，请确保使用 HTTPS 并配置合理的 API 鉴权机制。

**✨ 让 ComfyUI 的创作成果，一键触达你的业务系统！**
