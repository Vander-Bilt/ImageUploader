"""
ImageUploaderNode - 将 ComfyUI 生成的图片上传到外部 API
"""
import torch
import numpy as np
import requests
import io
import folder_paths
from PIL import Image

# 节点类
class ImageUploader:
    def __init__(self):
        self.base_url = None  # 可在节点界面配置
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # 接收 VAE Decode 输出的 IMAGE [B,H,W,C]
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
                # 可选：添加更多参数，如超时时间、重试次数等
            }
        }
    
    RETURN_TYPES = ("STRING", "JSON")  # 返回：1) URL 列表字符串 2) 原始响应 JSON
    RETURN_NAMES = ("urls", "response_data")
    FUNCTION = "upload_images"
    CATEGORY = "image/upload"
    OUTPUT_NODE = True  # 标记为输出节点，会在执行后显示结果

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        将单个 [H,W,C] 的 tensor 转换为 PIL.Image
        ComfyUI 的 IMAGE 是 [0,1] 范围的 float32，需要转 [0,255] uint8
        """
        # 确保在 CPU 上操作
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # [H,W,C] float32 [0,1] → [H,W,C] uint8 [0,255]
        np_image = (tensor.numpy() * 255.0).astype(np.uint8)
        
        # 创建 PIL Image (RGB 模式)
        pil_image = Image.fromarray(np_image, mode='RGB')
        return pil_image

    def pil_to_bytes(self, pil_image: Image.Image, format: str = 'PNG') -> bytes:
        """将 PIL.Image 转换为字节流"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=95)
        buffer.seek(0)
        return buffer.getvalue()

    def upload_single_image(self, api_url: str, image_bytes: bytes, content_type: str) -> dict:
        """调用后端 API 上传单个图片"""
        headers = {
            'Content-Type': content_type,
        }
        
        try:
            response = requests.post(
                api_url,
                data=image_bytes,
                headers=headers,
                timeout=300  # 5分钟超时，支持大文件
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

    def upload_images(self, images: torch.Tensor, api_base_url: str, upload_endpoint: str):
        """
        主函数：处理批量图片上传
        
        参数:
            images: torch.Tensor, shape [B, H, W, C], C=3, 值域 [0,1]
            api_base_url: str, 如 "http://localhost:3011"
            upload_endpoint: str, 如 "/upload-image-binary"
        
        返回:
            (urls字符串, 响应数据字典)
        """
        # 构建完整 API URL
        api_url = f"{api_base_url.rstrip('/')}/{upload_endpoint.lstrip('/')}"
        print(f"🚀 开始上传图片到: {api_url}")
        
        # 确保 images 是 4D: [B,H,W,C]
        if images.dim() == 3:
            images = images.unsqueeze(0)  # 单张图 [H,W,C] → [1,H,W,C]
        
        batch_size = images.shape[0]
        print(f"📦 待上传图片数量: {batch_size}")
        
        results = []
        urls = []
        
        for i in range(batch_size):
            print(f"⬆️  正在上传第 {i+1}/{batch_size} 张...")
            
            # 1. 提取单张图 [H,W,C]
            single_tensor = images[i]
            
            # 2. Tensor → PIL.Image
            pil_img = self.tensor_to_pil(single_tensor)
            
            # 3. PIL → bytes (PNG 格式)
            img_bytes = self.pil_to_bytes(pil_img, format='PNG')
            content_type = 'image/png'
            
            # 4. 调用 API 上传
            result = self.upload_single_image(api_url, img_bytes, content_type)
            results.append(result)
            
            if result['success']:
                url = result['data'].get('url', 'N/A')
                urls.append(url)
                print(f"✅ 上传成功: {url}")
            else:
                error_msg = result.get('error', '未知错误')
                urls.append(f"ERROR: {error_msg}")
                print(f"❌ 上传失败: {error_msg}")
        
        # 5. 准备返回值
        # 返回格式 1: URL 列表字符串（方便预览/日志）
        urls_str = "\n".join(urls)
        
        # 返回格式 2: 完整响应数据（方便后续节点处理）
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
    "ImageUploader": "📤 Upload Image to API",
}