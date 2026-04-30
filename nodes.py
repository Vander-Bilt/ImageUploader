"""
ImageUploaderNode - 将 ComfyUI 生成的图片上传到外部 API（支持特征向量 + 多标签）
"""
import torch
import numpy as np
import requests
import io
import json
import folder_paths
from PIL import Image

# 节点类
class ImageUploader:
    def __init__(self):
        self.base_url = None
        
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
            },
            "optional": {
                "features": ("CLIP_VISION_OUTPUT",),
                
                # 🔹 新增 2: 图片标签（支持多标签，逗号/换行分隔）
                "labels": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "每行一个标签，或用逗号分隔\ne.g. cat, outdoor, sunset\n或\n猫\n户外\n日落"
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

    # 🔹 特征向量预处理
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
    
        # 🔹 6. 可选压缩
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
                          labels: list = None, index: int = 0) -> dict:
        """调用后端 API 上传单个图片（支持 multipart + JSON 混合）"""
        
        # 方案: 使用 multipart/form-data 同时发送文件和元数据
        files = {
            'image': (f'image_{index}.png', image_bytes, content_type),
        }
        
        # 构建元数据字段
        metadata = {}
        if feature_vec is not None:
            metadata['feature_vector'] = feature_vec
        if labels is not None:
            metadata['labels'] = labels
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
                     features: torch.Tensor = None, labels: str = ""):
        """
        主函数：处理批量图片上传（支持特征向量 + 标签）
        """
        
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
        
        # 🔹 预处理特征向量
        feature_list = self.process_features(features) if features is not None else None
        if feature_list is not None:
            # 如果特征数量与图片数量不匹配，尝试广播或报错
            if len(feature_list) == 1 and batch_size > 1:
                import copy
                feature_list = [copy.deepcopy(feature_list[0]) for _ in range(batch_size)]
            elif len(feature_list) != batch_size:
                print(f"⚠️ 特征数量({len(feature_list)})与图片数量({batch_size})不匹配，将使用第一个特征")
                feature_list = [feature_list[0]] * batch_size
        
        # 🔹 解析标签
        label_list = self.parse_labels(labels, batch_size)
        
        results = []
        urls = []
        
        for i in range(batch_size):
            print(f"⬆️  正在上传第 {i+1}/{batch_size} 张...")
            
            # 1. 提取单张图
            single_tensor = images[i]
            pil_img = self.tensor_to_pil(single_tensor)
            img_bytes = self.pil_to_bytes(pil_img, format='PNG')
            
            # 2. 获取当前图片的特征和标签
            feat = feature_list[i] if feature_list else None
            lbl = label_list[i] if label_list else None
            
            # 3. 调用 API 上传
            result = self.upload_single_image(
                api_url, img_bytes, 'image/png',
                feature_vec=feat, labels=lbl, index=i
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