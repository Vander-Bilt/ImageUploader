"""
ImageUploaderNode - 将 ComfyUI 生成的图片上传到外部 API（支持特征向量 + 多标签）
支持内部提取 Chinese-CLIP 特征或使用外部传入特征
"""
import torch
import numpy as np
import requests
import io
import json
import folder_paths
from PIL import Image
import os

# 全局模型缓存
_CHINESE_CLIP_MODEL = None
_CHINESE_CLIP_PROCESSOR = None

# 节点类
class ImageUploader:
    def __init__(self):
        self.base_url = None
        self.clip_model = None
        self.clip_processor = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # [B,H,W,C], 值域 [0,1]
                "api_base_url": ("STRING", {
                    "default": "http://localhost:3011",
                    "multiline": False
                }),
                "upload_endpoint": ("STRING", {
                    "default": "/upload-image-binary",
                    "multiline": False
                }),
                # 🔹 新增：是否内部提取特征
                "innerExtract": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Use Chinese-CLIP",
                    "label_off": "Use External Features"
                }),
            },
            "optional": {
                "features": ("CLIP_VISION_OUTPUT",),
                
                # 🔹 新增 2: 扩展信息字段（JSON 字符串）
                "extra_info": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "placeholder": 'e.g. {"tags": [["tag1", "tag2"], ["tag3"]], "titles": ["title1", "title2"]}'
                }),
                
                # 🔹 新增 3: 总体信息字段（JSON 字符串）
                "batchInfo": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "placeholder": 'e.g. {"host"":"neoc", "category":"风景", "collection":"公园", "style":"真实"}'
                }),
                
                # 🔹 新增 4: Chinese-CLIP 模型配置（仅当 innerExtract=True 时生效）
                "clip_model_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": ""
                }),
                
                # 🔹 新增 5: Chinese-CLIP 模型 revision（可选，为空则不指定）
                "clip_model_revision": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "commit version"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("urls", "response_data")
    FUNCTION = "upload_images"
    CATEGORY = "image/upload"
    OUTPUT_NODE = True

    # ========== 原有工具方法保持不变 ==========
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.is_cuda:
            tensor = tensor.cpu()
        np_image = (tensor.numpy() * 255.0).astype(np.uint8)
        channels = np_image.shape[-1] if np_image.ndim == 3 else 3
        if channels == 4:
            pil_image = Image.fromarray(np_image, mode='RGBA')
        elif channels == 3:
            pil_image = Image.fromarray(np_image, mode='RGB')
        elif channels == 1:
            pil_image = Image.fromarray(np_image.squeeze(), mode='L')
        else:
            pil_image = Image.fromarray(np_image[:, :, :3], mode='RGB')
        return pil_image

    def pil_to_bytes(self, pil_image: Image.Image, format: str = 'PNG') -> bytes:
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=95)
        buffer.seek(0)
        return buffer.getvalue()

    # 🔹 新增：加载 Chinese-CLIP 模型（单例模式）
    def load_chinese_clip_model(self, model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14", 
                                revision: str = ""):
        """
        加载 Chinese-CLIP 模型，使用 safetensors 版本避开 torch.load 安全限制
        
        Args:
            model_name: 模型名称
            revision: 模型版本（可选，为空则使用默认版本）
        """
        global _CHINESE_CLIP_MODEL, _CHINESE_CLIP_PROCESSOR
        
        # 检查缓存
        if _CHINESE_CLIP_MODEL is not None and _CHINESE_CLIP_PROCESSOR is not None:
            print("✅ 使用缓存的 Chinese-CLIP 模型")
            return _CHINESE_CLIP_MODEL, _CHINESE_CLIP_PROCESSOR
        
        try:
            print(f"📥 正在加载 Chinese-CLIP 模型: {model_name}")
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            
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
            
            # 加载 Processor
            _CHINESE_CLIP_PROCESSOR = ChineseCLIPProcessor.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # 加载 Model
            _CHINESE_CLIP_MODEL = ChineseCLIPModel.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # 移到 GPU（如果可用）
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _CHINESE_CLIP_MODEL = _CHINESE_CLIP_MODEL.to(device)
            _CHINESE_CLIP_MODEL.eval()
            
            revision_info = f"revision={revision}" if revision else "默认版本"
            print(f"✅ Chinese-CLIP 模型加载成功 (device: {device}, {revision_info}, safetensors: True)")
            return _CHINESE_CLIP_MODEL, _CHINESE_CLIP_PROCESSOR
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Chinese-CLIP 模型加载失败: {error_msg}")
            
            # 🔹 提供详细的故障排查建议
            if "torch.load" in error_msg and "v2.6" in error_msg:
                print("\n💡 解决方案:")
                print("   1. 尝试指定包含 safetensors 的 revision")
                print("      例如: cc2eeb803885adc11690654a6e55fde2feeb7420")
                print("   2. 请检查 transformers 版本: pip install --upgrade transformers>=4.36.0")
                print("   3. 或手动清理缓存后重试:")
                print(f"      rm -rf ~/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14")
            elif "404" in error_msg or "Not Found" in error_msg:
                print("\n💡 错误: 找不到指定的 revision")
                print("   请检查网络连接或尝试:")
                print(f"      huggingface-cli download {model_name}")
                if revision:
                    print(f"      或指定 revision: --revision {revision}")
            
            raise e

    # 🔹 新增：使用 Chinese-CLIP 提取图像特征
    def extract_features_with_chinese_clip(self, images: torch.Tensor, model_name: str, revision: str = ""):
        """
        使用 Chinese-CLIP 提取图像特征向量
        Args:
            images: [B,H,W,C] tensor, 值域 [0,1]
            model_name: 模型名称
            revision: 模型版本（可选）
        Returns:
            List of feature vectors (normalized)
        """
        print("🎯 开始使用 Chinese-CLIP 提取特征...")
        
        # 1. 加载模型
        model, processor = self.load_chinese_clip_model(model_name, revision)
        device = next(model.parameters()).device
        
        # 2. 转换 PIL 图像
        pil_images = []
        for i in range(images.shape[0]):
            pil_img = self.tensor_to_pil(images[i])
            pil_images.append(pil_img)
        
        # 3. 使用 processor 处理图像（自动 resize/normalize）
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 4. 提取特征（禁用梯度）
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # L2 归一化（Cosine Similarity 必须）
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        
        # 5. 移到 CPU 并转 list
        image_features = image_features.cpu()
        feat_list = image_features.tolist()
        
        print(f"✅ Chinese-CLIP 特征提取完成: {len(feat_list)} 张图, 维度: {len(feat_list[0])}")
        return feat_list

    # 🔹 特征向量预处理（兼容外部传入的 features）
    def process_features(self, features, pool_mode: str = "cls", compress: bool = True):
        """增强版：兼容 comfy.clip_vision.Output 对象 + dict + Tensor"""
        print(f"🔍 [process_features] 1. 输入类型: {type(features)}")
    
        # 🔹 1. 优先处理 ComfyUI 原生 Output 对象（鸭类型检查）
        if hasattr(features, "image_embeds"):
            print("✅ 识别到 CLIP_VISION_OUTPUT，提取 image_embeds")
            features = features.image_embeds
        elif hasattr(features, "last_hidden_state"):
            print("✅ 识别到 CLIP_VISION_OUTPUT，提取 last_hidden_state")
            features = features.last_hidden_state
        elif isinstance(features, dict):
            # 兼容某些自定义节点返回的 dict
            if "image_embeds" in features:
                features = features["image_embeds"]
            elif "last_hidden_state" in features:
                features = features["last_hidden_state"]
            else:
                print(f"⚠️ dict 中缺少特征字段，可用键: {list(features.keys())}")
                return None
        elif not isinstance(features, torch.Tensor):
            print(f"❌ 不支持的类型: {type(features)}")
            return None
    
        # 🔹 2. 确保是 Tensor
        if features is None or not isinstance(features, torch.Tensor):
            print("❌ 提取后仍不是 Tensor，跳过")
            return None
    
        print(f"🔍 [process_features] 2. Tensor: shape={features.shape}, dtype={features.dtype}, device={features.device}")
    
        # 🔹 3. 移到 CPU
        if features.is_cuda:
            features = features.cpu()
    
        # 🔹 4. 确保维度为 [B, D]
        if features.dim() == 1:
            features = features.unsqueeze(0)
            print(f"✅ 1D→2D: {features.shape}")
        elif features.dim() == 3:
            if pool_mode == "cls":
                features = features[:, 0, :]
            else:  # mean
                features = features.mean(dim=1)
            print(f"✅ 3D 池化完成 ({pool_mode}): {features.shape}")
    
        # 🔹 5. 清洗 NaN/Inf（防止 JSON 序列化失败）
        if features.dtype.is_floating_point:
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
        # 🔹 6. 归一化 (Cosine Similarity 必须)
        features = features / torch.linalg.norm(features, dim=-1, keepdim=True)
        print("✅ 向量归一化完成")

        # 🔹 7. 可选压缩
        if compress and features.dtype == torch.float32:
            features = features.half()
            print("✅ 压缩: float32 → float16")
    
        # 🔹 7. 转 list
        feat_list = features.tolist()
        print(f"✅ 转 list 完成: {len(feat_list)} 张图, 维度: {len(feat_list[0]) if feat_list else 0}")
        return feat_list

    # 🔹 新增：标签解析（支持批量 & 多格式）
    def parse_labels(self, labels_str: str, batch_size: int) -> list:
        """
        解析标签字符串，返回长度为 batch_size 的标签列表
        支持格式:
          - "tag1,tag2,tag3" → 所有图片共用这组标签
          - "tag1\ntag2\ntag3" → 每行对应一张图片的标签（可含逗号）
          - "a,b|c,d|e,f" → 用 | 分隔每张图的标签组
        """
        if not labels_str or not labels_str.strip():
            return [None] * batch_size
            
        lines = [l.strip() for l in labels_str.strip().split('\n') if l.strip()]
        
        # 情况1: 只有一行 → 所有图片共用标签
        if len(lines) == 1:
            # 支持逗号或空格分隔
            tags = [t.strip() for t in lines[0].replace('，', ',').split(',') if t.strip()]
            return [tags] * batch_size
            
        # 情况2: 多行 → 尝试匹配图片数量
        if len(lines) == batch_size:
            result = []
            for line in lines:
                tags = [t.strip() for t in line.replace('，', ',').split(',') if t.strip()]
                result.append(tags if tags else None)
            return result
            
        # 情况3: 用 | 显式分隔每张图的标签
        if '|' in labels_str:
            groups = [g.strip() for g in labels_str.split('|') if g.strip()]
            if len(groups) == batch_size:
                result = []
                for g in groups:
                    tags = [t.strip() for t in g.replace('，', ',').split(',') if t.strip()]
                    result.append(tags if tags else None)
                return result
                
        # fallback: 所有图片共用第一行解析的标签
        tags = [t.strip() for t in lines[0].replace('，', ',').split(',') if t.strip()]
        return [tags] * batch_size

    def upload_single_image(self, api_url: str, image_bytes: bytes, 
                          content_type: str, feature_vec: list = None, 
                          tags: list = None, title: str = None,
                          batch_info: dict = None,
                          index: int = 0,
                          msg: str = None) -> dict:
        """调用后端 API 上传单个图片（支持 multipart + JSON 混合）"""
        
        # 方案: 使用 multipart/form-data 同时发送文件和元数据
        files = {
            'image': (f'image_{index}.png', image_bytes, content_type),
        }

        print(f"🔍 [DEBUG] 图片title: {title} tags: {tags}")
        
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
            metadata['msg'] = msg
        metadata['index'] = index  # 图片在 batch 中的序号
        
        
        # metadata = {
#             'feature_vector': [0.1, 0.2, 0.3, 0.4, 0.5],  # 手动写死小向量
#             'labels': ['test', 'debug'],
#             'index': index
#         }
        
        # 在构建 metadata 后，发送请求前添加：
        print(f"🔍 [DEBUG] 准备发送 metadata: {json.dumps(metadata, ensure_ascii=False)[:200]}...")  # 打印前200字符
        if 'feature_vector' in metadata:
            feat = metadata['feature_vector']
            print(f"📊 [DEBUG] 特征向量类型: {type(feat)}, 长度: {len(feat) if isinstance(feat, list) else 'N/A'}")
            if isinstance(feat, list) and len(feat) > 0:
                print(f"📊 [DEBUG] 前5个值: {feat[:5]}")
        
        
        data = {
            'metadata': json.dumps(metadata, ensure_ascii=False)
        }
        
        try:
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=300
            )
            response.raise_for_status()
            return {
                'success': True,
                'data': response.json(),
                'status_code': response.status_code
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'data': None
            }

    def upload_images(self, images: torch.Tensor, api_base_url: str, upload_endpoint: str,
                     innerExtract: bool = False,
                     features: torch.Tensor = None, extra_info: str = "{}", batchInfo: str = "{}",
                     clip_model_name: str = "OFA-Sys/chinese-clip-vit-large-patch14",
                     clip_model_revision: str = ""):
        """
        主函数：处理批量图片上传（支持特征向量 + 扩展信息 + 总体信息）
        
        Args:
            innerExtract: 如果为 True，使用 Chinese-CLIP 内部提取特征，忽略 features 参数
            clip_model_name: Chinese-CLIP 模型名称（仅当 innerExtract=True 时生效）
            clip_model_revision: Chinese-CLIP 模型版本（可选，为空则使用默认版本）
        """
        
        # 🔧 解析总体信息 (JSON)
        import json
        try:
            batch_info_dict = json.loads(batchInfo) if batchInfo else {}
        except Exception as e:
            print(f"⚠️ batchInfo JSON 解析失败: {e}")
            batch_info_dict = {"raw_batch_info": batchInfo}
            
        # 🔧 解析扩展信息 (JSON)
        try:
            extra_info_dict = json.loads(extra_info) if extra_info else {}
        except Exception as e:
            print(f"⚠️ extra_info JSON 解析失败: {e}")
            extra_info_dict = {}

        tags_list = extra_info_dict.get("tags", [])
        titles_list = extra_info_dict.get("titles", [])
        
        # 🔧 类型适配：支持 CLIP_VISION_OUTPUT
        if isinstance(features, dict):
            if "image_embeds" in features:
                features = features["image_embeds"]
            elif "last_hidden_state" in features and features["last_hidden_state"].dim() == 3:
                # 全局平均池化
                features = features["last_hidden_state"].mean(dim=1)
            else:
                print(f"⚠️ 无法从 CLIP 输出提取特征，跳过")
                features = None
        
        api_url = f"{api_base_url.rstrip('/')}/{upload_endpoint.lstrip('/')}"
        print(f"🚀 开始上传图片到: {api_url}")
        
        # 确保 images 是 4D: [B,H,W,C]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        batch_size = images.shape[0]
        print(f"📦 待上传图片数量: {batch_size}")
        
        # 🔹 特征提取逻辑分支
        feature_list = None
        extract_method = ""  # 记录特征提取方法
        
        if innerExtract:
            # 🔹 分支1: 使用 Chinese-CLIP 内部提取
            print("🎯 模式: 内部提取 (Chinese-CLIP)")
            extract_method = "Chinese-CLIP内部提取"
            try:
                feature_list = self.extract_features_with_chinese_clip(images, clip_model_name, clip_model_revision)
            except Exception as e:
                print(f"❌ Chinese-CLIP 特征提取失败: {e}")
                print("⚠️ 将继续上传但不包含特征向量")
                feature_list = None
                extract_method = "Chinese-CLIP内部提取失败"
        else:
            # 🔹 分支2: 使用外部传入的 features
            print("🎯 模式: 使用外部传入特征")
            extract_method = "外部传入特征"
            if features is not None:
                feature_list = self.process_features(features)
                if feature_list is not None:
                    # 如果特征数量与图片数量不匹配，尝试广播或报错
                    if len(feature_list) == 1 and batch_size > 1:
                        import copy
                        feature_list = [copy.deepcopy(feature_list[0]) for _ in range(batch_size)]
                    elif len(feature_list) != batch_size:
                        print(f"⚠️ 特征数量({len(feature_list)})与图片数量({batch_size})不匹配，将使用第一个特征")
                        feature_list = [feature_list[0]] * batch_size
                else:
                    extract_method = "外部特征处理失败"
            else:
                print("⚠️ 未提供 features 参数，将不上传特征向量")
                extract_method = "未提供外部特征"
        
        # 构建消息字符串
        msg = f"特征提取方式: {extract_method}"
        if innerExtract and clip_model_name:
            msg += f" | 模型: {clip_model_name}"
            if clip_model_revision:
                msg += f" | 版本: {clip_model_revision}"
        msg += f" | 批次大小: {batch_size}"
        
        print(f"📝 上传消息: {msg}")
        
        results = []
        urls = []
        
        for i in range(batch_size):
            print(f"⬆️  正在上传第 {i+1}/{batch_size} 张...")
            
            # 1. 提取单张图
            single_tensor = images[i]
            pil_img = self.tensor_to_pil(single_tensor)
            img_bytes = self.pil_to_bytes(pil_img, format='PNG')
            
            # 2. 获取当前图片的特征和扩展信息
            feat = feature_list[i] if feature_list else None
            
            # 从 extra_info 获取当前图片的 tags 和 title
            current_tags = tags_list[i] if i < len(tags_list) else None
            current_title = titles_list[i] if i < len(titles_list) else None
            
            # 3. 调用 API 上传
            result = self.upload_single_image(
                api_url, img_bytes, 'image/png',
                feature_vec=feat, 
                tags=current_tags,
                title=current_title,
                batch_info=batch_info_dict, 
                index=i,
                msg=msg  # 传递消息
            )
            results.append(result)
            
            if result['success']:
                url = result['data'].get('url', 'N/A')
                urls.append(url)
                print(f"✅ 上传成功: {url}")
            else:
                error_msg = result.get('error', '未知错误')
                urls.append(f"ERROR: {error_msg}")
                print(f"❌ 上传失败: {error_msg}")
        
        # 4. 准备返回值
        urls_str = "\n".join(urls)
        response_data = {
            'total': batch_size,
            'success_count': sum(1 for r in results if r['success']),
            'results': results,
            'urls': urls
        }
        
        print(f"🎉 上传完成: {response_data['success_count']}/{batch_size} 成功")
        return (urls_str, response_data)


# 节点注册映射
NODE_CLASS_MAPPINGS = {
    "ImageUploader": ImageUploader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageUploader": "📤 Upload Image to API (+Feat+Tags)",
}