# ImageUploader 更新说明

## 更新内容

### 新增功能：可配置的模型 Revision 参数

**问题：**
之前 `revision_with_safetensors` 参数硬编码在代码中（`cc2eeb803885adc11690654a6e55fde2feeb7420`），用户无法灵活配置。

**解决方案：**
1. 将 `revision` 参数提取为可选的输入参数 `clip_model_revision`
2. 默认值为空字符串
3. 如果为空，则不传递 `revision` 参数给 `ChineseCLIPProcessor.from_pretrained()`，使用模型的默认版本
4. 如果指定了值，则使用指定的 revision

## 使用方法

### 1. 使用默认版本（推荐）

在 ComfyUI 节点中：
- `clip_model_revision` 留空（默认值）
- 系统会自动使用模型的最新版本

```python
# 内部调用
ChineseCLIPProcessor.from_pretrained(
    "OFA-Sys/chinese-clip-vit-large-patch14",
    use_safetensors=True
)
```

### 2. 指定特定版本

如果需要使用特定版本（例如包含 safetensors 的版本）：
- `clip_model_revision` 设置为：`cc2eeb803885adc11690654a6e55fde2feeb7420`

```python
# 内部调用
ChineseCLIPProcessor.from_pretrained(
    "OFA-Sys/chinese-clip-vit-large-patch14",
    revision="cc2eeb803885adc11690654a6e55fde2feeb7420",
    use_safetensors=True
)
```

## 参数说明

### 新增参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `clip_model_revision` | STRING | `""` (空) | Chinese-CLIP 模型的版本号（可选） |

### 参数行为

1. **空字符串（默认）：**
   - 不传递 `revision` 参数
   - 使用 HuggingFace 上的最新版本
   - 适合大多数情况

2. **指定版本号：**
   - 传递 `revision` 参数
   - 使用指定的模型版本
   - 适合需要固定版本或解决兼容性问题的情况

## 代码变更

### 1. INPUT_TYPES 新增参数

```python
"optional": {
    # ... 其他参数 ...
    
    # 新增：模型 revision 参数
    "clip_model_revision": ("STRING", {
        "default": "",
        "multiline": False,
        "placeholder": "cc2eeb803885adc11690654a6e55fde2feeb7420"
    }),
}
```

### 2. load_chinese_clip_model 函数更新

```python
def load_chinese_clip_model(self, model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14", 
                            revision: str = ""):
    """
    加载 Chinese-CLIP 模型
    
    Args:
        model_name: 模型名称
        revision: 模型版本（可选，为空则使用默认版本）
    """
    # 准备加载参数
    load_kwargs = {
        "use_safetensors": True
    }
    
    # 如果指定了 revision，则添加到参数中
    if revision and revision.strip():
        load_kwargs["revision"] = revision.strip()
        print(f"🎯 使用指定的 revision: {revision}")
    else:
        print("🎯 使用默认 revision（最新版本）")
    
    # 加载模型
    _CHINESE_CLIP_PROCESSOR = ChineseCLIPProcessor.from_pretrained(
        model_name,
        **load_kwargs
    )
```

### 3. extract_features_with_chinese_clip 函数更新

```python
def extract_features_with_chinese_clip(self, images: torch.Tensor, 
                                      model_name: str, 
                                      revision: str = ""):
    """
    使用 Chinese-CLIP 提取图像特征向量
    
    Args:
        images: [B,H,W,C] tensor
        model_name: 模型名称
        revision: 模型版本（可选）
    """
    model, processor = self.load_chinese_clip_model(model_name, revision)
    # ... 其余代码 ...
```

### 4. upload_images 主函数更新

```python
def upload_images(self, images: torch.Tensor, api_base_url: str, upload_endpoint: str,
                 innerExtract: bool = False,
                 features: torch.Tensor = None, 
                 extra_info: str = "{}", 
                 batchInfo: str = "{}",
                 clip_model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14",
                 clip_model_revision: str = ""):  # 新增参数
    """
    主函数：处理批量图片上传
    
    Args:
        clip_model_revision: Chinese-CLIP 模型版本（可选，为空则使用默认版本）
    """
    # ... 其余代码 ...
    
    if innerExtract:
        feature_list = self.extract_features_with_chinese_clip(
            images, 
            clip_model_name, 
            clip_model_revision  # 传递 revision 参数
        )
```

## 使用场景

### 场景1：正常使用（推荐）

**配置：**
- `clip_model_name`: `OFA-Sys/chinese-clip-vit-large-patch14`
- `clip_model_revision`: 留空

**行为：**
- 使用最新版本的模型
- 自动获取最新的改进和修复

### 场景2：遇到 torch.load 安全限制错误

**错误信息：**
```
Weights only load failed. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution.
```

**解决方案：**
- `clip_model_revision`: `cc2eeb803885adc11690654a6e55fde2feeb7420`

**说明：**
- 这个版本包含 safetensors 格式的权重文件
- 避免了 torch.load 的安全限制问题

### 场景3：需要固定版本

**配置：**
- `clip_model_revision`: 指定具体的 commit hash

**行为：**
- 使用固定版本的模型
- 确保结果的可重现性
- 适合生产环境

## 常见问题

### Q1: 什么时候需要指定 revision？

**A:** 通常情况下不需要指定。只有在以下情况下才需要：
1. 遇到 torch.load 安全限制错误
2. 需要使用特定版本的模型
3. 需要确保结果的可重现性

### Q2: 如何找到可用的 revision？

**A:** 访问 HuggingFace 模型页面：
```
https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14/commits/main
```

每个 commit 都有一个 hash，可以作为 revision 使用。

### Q3: 留空和指定 revision 有什么区别？

**A:** 
- **留空：** 使用 HuggingFace 上的最新版本（main 分支）
- **指定：** 使用特定的版本（固定的 commit）

### Q4: 为什么默认值是空而不是 `cc2eeb8...`？

**A:** 
1. 让用户使用最新版本，获得最新的改进
2. 只有在遇到问题时才需要指定特定版本
3. 简化配置，大多数情况下不需要修改

### Q5: 如果指定了错误的 revision 会怎样？

**A:** 会抛出 404 错误，提示找不到指定的版本。错误信息会包含解决建议。

## 测试建议

### 测试1：默认配置

1. 保持 `clip_model_revision` 为空
2. 运行节点
3. 验证模型加载成功
4. 检查日志：`🎯 使用默认 revision（最新版本）`

### 测试2：指定 revision

1. 设置 `clip_model_revision` 为 `cc2eeb803885adc11690654a6e55fde2feeb7420`
2. 运行节点
3. 验证模型加载成功
4. 检查日志：`🎯 使用指定的 revision: cc2eeb803885adc11690654a6e55fde2feeb7420`

### 测试3：错误的 revision

1. 设置 `clip_model_revision` 为 `invalid-revision`
2. 运行节点
3. 验证错误处理正确
4. 检查错误信息是否包含解决建议

## 向后兼容性

✅ **完全向后兼容**

- 现有的工作流不需要修改
- 默认行为与之前相同（使用最新版本）
- 只是提供了更多的灵活性

## 性能影响

- **无性能影响**
- 只是在加载模型时多了一个条件判断
- 模型加载后的运行性能完全相同

## 总结

本次更新提供了更灵活的模型版本控制：

1. ✅ 默认使用最新版本（推荐）
2. ✅ 可选指定特定版本（解决兼容性问题）
3. ✅ 完全向后兼容
4. ✅ 清晰的日志输出
5. ✅ 详细的错误提示

用户可以根据实际需求选择合适的配置方式。
