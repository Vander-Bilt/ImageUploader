# 消息传递功能说明

## 功能概述

在图片上传过程中，将特征提取方式等信息通过 `msg` 字段从 ComfyUI 节点传递到后端服务器。

## 实现细节

### 1. ImageUploader/nodes.py

#### 1.1 构建消息内容

在 `upload_images` 函数中，根据特征提取方式构建消息：

```python
# 构建消息字符串
msg = f"特征提取方式: {extract_method}"
if innerExtract and clip_model_name:
    msg += f" | 模型: {clip_model_name}"
    if clip_model_revision:
        msg += f" | 版本: {clip_model_revision}"
msg += f" | 批次大小: {batch_size}"

print(f"📝 上传消息: {msg}")
```

**消息格式示例：**

1. **使用 Chinese-CLIP 内部提取：**
   ```
   特征提取方式: Chinese-CLIP内部提取 | 模型: OFA-Sys/chinese-clip-vit-large-patch14 | 版本: cc2eeb803885adc11690654a6e55fde2feeb7420 | 批次大小: 4
   ```

2. **使用外部传入特征：**
   ```
   特征提取方式: 外部传入特征 | 批次大小: 4
   ```

3. **未提供特征：**
   ```
   特征提取方式: 未提供外部特征 | 批次大小: 4
   ```

4. **特征提取失败：**
   ```
   特征提取方式: Chinese-CLIP内部提取失败 | 模型: OFA-Sys/chinese-clip-vit-large-patch14 | 批次大小: 4
   ```

#### 1.2 传递消息到 upload_single_image

```python
result = self.upload_single_image(
    api_url, img_bytes, 'image/png',
    feature_vec=feat, 
    tags=current_tags,
    title=current_title,
    batch_info=batch_info_dict, 
    index=i,
    msg=msg  # 传递消息
)
```

#### 1.3 添加到 metadata

在 `upload_single_image` 函数中：

```python
def upload_single_image(self, api_url: str, image_bytes: bytes, 
                      content_type: str, feature_vec: list = None, 
                      tags: list = None, title: str = None,
                      batch_info: dict = None,
                      index: int = 0,
                      msg: str = None) -> dict:  # 新增 msg 参数
    
    # 构建元数据字段
    metadata = {}
    if feature_vec is not None:
        metadata['feature_vector'] = feature_vec
    if tags is not None:
        metadata['tags'] = tags
    if title is not None:
        metadata['title'] = title
    if batch_info is not None:
        metadata['batch_info'] = batch_info
    if msg is not None:
        metadata['msg'] = msg  # 添加 msg 字段
    metadata['index'] = index
```

### 2. PhotoCake/Uploader/app.js

#### 2.1 接收并解析 msg

在 `/upload-image-binary` 接口中：

```javascript
// 🔹 2. 解析 metadata
let clipVector = null;
let batchInfo = {};
let imageIndex = 0;
let tags = [];
let title = null;
let msg = null;  // 新增：接收消息字段

if (req.body.metadata) {
  try {
    const metadata = JSON.parse(req.body.metadata);
    clipVector = metadata.feature_vector || null;
    batchInfo = metadata.batch_info || {};
    imageIndex = metadata.index || 0;
    tags = metadata.tags || [];
    title = metadata.title || null;
    msg = metadata.msg || null;  // 新增：提取msg字段
    
    console.log(`📦 元数据解析成功: 索引=${imageIndex}, 标题="${title || '无'}", 标签=[${tags.join(', ')}]`);
    if (clipVector) console.log(`📊 收到特征向量 (维度: ${clipVector.length})`);
    if (msg) console.log(`💬 消息: ${msg}`);  // 新增：打印msg
  } catch (e) {
    console.warn('⚠️ metadata JSON 解析失败:', e.message);
  }
}
```

## 数据流程

```
ComfyUI节点 (nodes.py)
    ↓
1. 根据 innerExtract 判断特征提取方式
    ↓
2. 构建 msg 字符串
    ↓
3. 调用 upload_single_image(msg=msg)
    ↓
4. 将 msg 添加到 metadata
    ↓
5. 通过 requests.post 发送到后端
    ↓
后端服务器 (app.js)
    ↓
6. 解析 req.body.metadata
    ↓
7. 提取 msg 字段
    ↓
8. console.log 打印消息
```

## 测试方法

### 测试1：Chinese-CLIP 内部提取

**ComfyUI 节点配置：**
- `innerExtract`: True (勾选)
- `clip_model_name`: `OFA-Sys/chinese-clip-vit-large-patch14`
- `clip_model_revision`: `cc2eeb803885adc11690654a6e55fde2feeb7420`

**预期输出（nodes.py）：**
```
📝 上传消息: 特征提取方式: Chinese-CLIP内部提取 | 模型: OFA-Sys/chinese-clip-vit-large-patch14 | 版本: cc2eeb803885adc11690654a6e55fde2feeb7420 | 批次大小: 1
```

**预期输出（app.js）：**
```
💬 消息: 特征提取方式: Chinese-CLIP内部提取 | 模型: OFA-Sys/chinese-clip-vit-large-patch14 | 版本: cc2eeb803885adc11690654a6e55fde2feeb7420 | 批次大小: 1
```

### 测试2：外部传入特征

**ComfyUI 节点配置：**
- `innerExtract`: False (不勾选)
- `features`: 连接 CLIP Vision Encode 节点

**预期输出（nodes.py）：**
```
📝 上传消息: 特征提取方式: 外部传入特征 | 批次大小: 1
```

**预期输出（app.js）：**
```
💬 消息: 特征提取方式: 外部传入特征 | 批次大小: 1
```

### 测试3：未提供特征

**ComfyUI 节点配置：**
- `innerExtract`: False (不勾选)
- `features`: 不连接

**预期输出（nodes.py）：**
```
📝 上传消息: 特征提取方式: 未提供外部特征 | 批次大小: 1
```

**预期输出（app.js）：**
```
💬 消息: 特征提取方式: 未提供外部特征 | 批次大小: 1
```

## 日志示例

### 完整的上传日志（app.js）

```
🚀 [2024-01-15T10:30:45.123Z] 开始处理新上传请求...
📁 收到文件: image_0.png -> 1705315845123.png (2.45 MB)
🖼️  图片信息: 1024x768, 格式: png
📦 元数据解析成功: 索引=0, 标题="测试图片", 标签=[风景, 自然]
📊 收到特征向量 (维度: 768)
💬 消息: 特征提取方式: Chinese-CLIP内部提取 | 模型: OFA-Sys/chinese-clip-vit-large-patch14 | 版本: cc2eeb803885adc11690654a6e55fde2feeb7420 | 批次大小: 1
🪄  正在生成缩略图: 1705315845123_thumb.webp...
✅ 缩略图生成完毕: 500x375
📂 分类: 风景 (ID: 1)
📚 合集: 自然风光 (ID: 1)
💾 图片已存入数据库, ID: 123
🏷️  已关联 2 个标签
✨ 上传处理圆满完成! URL: http://localhost:3011/uploads/1705315845123.png
```

## 扩展建议

### 1. 将 msg 存入数据库

如果需要持久化这些信息，可以在数据库中添加字段：

```sql
ALTER TABLE images ADD COLUMN upload_msg TEXT;
```

然后在 app.js 中：

```javascript
const insertQuery = `
  INSERT INTO images (
    filename, host, filename_thumbnail, title, 
    width, height, thumbnail_width, thumbnail_height, 
    file_size, format, category_id, collection_id, 
    clip_vector, upload_msg, status
  )
  VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, 1)
  ...
`;

const imgRes = await client.query(insertQuery, [
  filename, host, thumbFilename, title,
  width, height, thumbInfo.width, thumbInfo.height,
  fileSize, format, categoryId, collectionId,
  clipVector ? pgvector.toSql(clipVector) : null,
  msg  // 添加 msg 参数
]);
```

### 2. 返回给前端

可以在响应中包含 msg：

```javascript
res.json({
  success: true,
  id: imageId,
  filename,
  thumbnail_url: thumbFilename,
  url: finalUrl,
  msg: msg  // 返回消息
});
```

### 3. 添加更多信息

可以在 msg 中添加更多有用的信息：

```python
msg = f"特征提取方式: {extract_method}"
msg += f" | 批次大小: {batch_size}"
msg += f" | 上传时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
msg += f" | 图片尺寸: {images.shape[1]}x{images.shape[2]}"
if feature_list:
    msg += f" | 特征维度: {len(feature_list[0])}"
```

## 故障排查

### 问题1：后端没有打印 msg

**检查项：**
1. 确认 nodes.py 中 msg 变量已正确构建
2. 确认 upload_single_image 调用时传递了 msg 参数
3. 确认 metadata 中包含了 msg 字段
4. 检查后端日志，确认 metadata 解析成功

### 问题2：msg 内容不正确

**检查项：**
1. 确认 extract_method 变量赋值正确
2. 确认 innerExtract 参数传递正确
3. 检查 nodes.py 中的日志输出

### 问题3：msg 为 null

**可能原因：**
1. nodes.py 中 msg 变量未定义
2. upload_single_image 未传递 msg 参数
3. metadata 构建时未添加 msg 字段

## 总结

通过这个功能，我们可以：

1. ✅ 追踪每张图片的特征提取方式
2. ✅ 记录使用的模型和版本
3. ✅ 便于调试和问题排查
4. ✅ 为后续的数据分析提供依据

这个功能为图片上传流程提供了更好的可观测性和可追溯性。
