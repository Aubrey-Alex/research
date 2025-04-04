import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionImg2ImgPipeline
from safetensors.torch import load_file
from PIL import Image
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def verify_image(image_path):
    """验证图像是否存在且可以正确加载"""
    if not os.path.exists(image_path):
        raise ValueError(f"图像文件不存在: {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((512, 512), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        raise ValueError(f"无法加载图像: {str(e)}")

# 加载模型
def load_model(model_type="hyper-sd"):
    try:
        # 根据模型类型选择不同的管道
        if model_type == "base":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to("cuda")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            ).to("cuda")
        
        # 设置调度器
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        
        # 根据模型类型加载不同的 LoRA 权重
        if model_type == "hyper-sd":
            state_dict = load_file("TSCD/LoRA/hypersd-sd1-5-1-step-lora/pytorch_lora_weights.safetensors")
            pipe.unet.load_state_dict(state_dict, strict=False)
        elif model_type == "lcm":
            state_dict = load_file("TSCD/LoRA/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors")
            pipe.unet.load_state_dict(state_dict, strict=False)
        
        pipe.enable_vae_slicing()
        return pipe
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

# 图像质量评估
def calculate_metrics(original_img, reconstructed_img):
    # 确保图像大小相同
    if original_img.shape != reconstructed_img.shape:
        reconstructed_img = cv2.resize(reconstructed_img, (original_img.shape[1], original_img.shape[0]))
    
    # PSNR
    mse = np.mean((original_img - reconstructed_img) ** 2)
    if mse == 0:
        return float('inf'), 1.0
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM
    ssim_score = ssim(original_img, reconstructed_img, multichannel=True, channel_axis=2)
    
    return psnr, ssim_score

# 使用潜变量方式的推理
def inference_with_latents(pipe, original_image, steps=1, model_type="hyper-sd"):
    # 编码为潜变量
    with torch.no_grad():
        image = pipe.image_processor.preprocess(original_image)
        image = image.to(device="cuda", dtype=torch.float16)
        latents = pipe.vae.encode(image).latent_dist.sample() * 0.18215
    
    # 前向加噪（扩散）
    noise = torch.randn_like(latents)
    pipe.scheduler.set_timesteps(steps)
    timestep = pipe.scheduler.timesteps[0]
    noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)
    
    # 反向去噪（生成）
    with torch.no_grad():
        if model_type == "lcm":
            # LCM: 8步，提高质量
            output = pipe(
                prompt="a clear photo of university gate with chinese characters, sharp details",
                negative_prompt="blur, low quality",  # 添加简单的负面提示词
                latents=noisy_latents,
                num_inference_steps=steps,
                guidance_scale=1.5,  # 增加引导比例以提高质量
            )
        elif model_type == "hyper-sd":
            # Hyper-SD: 快速且高质量
            output = pipe(
                prompt="a clear photo of university gate with chinese characters, sharp",
                negative_prompt="blur, low quality",
                latents=noisy_latents,
                num_inference_steps=steps,
                guidance_scale=2.0,
            )
        else:
            output = pipe(
                prompt="",
                latents=noisy_latents,
                num_inference_steps=steps,
                guidance_scale=1.0,
            )
    
    reconstructed_image = output.images[0]
    return reconstructed_image

# 使用 img2img 方式的推理
def inference_with_img2img(pipe, original_image, steps=25):
    with torch.no_grad():
        output = pipe(
            prompt="a professional architectural photo of university gate with chinese characters, ultra detailed, sharp focus",
            negative_prompt="blur, noise, low quality, distortion, deformation, watermark",
            image=original_image,
            strength=0.4,  # 增加强度以提高质量
            num_inference_steps=steps,
            guidance_scale=8.5,  # 增加引导比例以提高质量
        )
    
    return output.images[0]

# 完整前向+反向流程
def test_pipeline(pipe, input_image_path, model_type="base", steps=1):
    try:
        original_image = verify_image(input_image_path)
        original_img_np = np.array(original_image)
        
        start_time = time.time()
        
        if model_type == "base":
            reconstructed_image = inference_with_img2img(pipe, original_image, steps)
        else:
            reconstructed_image = inference_with_latents(pipe, original_image, steps, model_type)
            
        latency = time.time() - start_time
        
        reconstructed_img_np = np.array(reconstructed_image)
        psnr, ssim_score = calculate_metrics(original_img_np, reconstructed_img_np)
        
        return reconstructed_image, latency, psnr, ssim_score
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise

# 主函数    
if __name__ == "__main__":
    try:
        # 测试配置
        input_image_path = "TSCD/image/ZJU1.jpg"
        
        # 验证输入图像
        if not os.path.exists(input_image_path):
            raise ValueError(f"输入图像不存在: {input_image_path}")
        
        # 定义要测试的模型和它们的步数
        model_configs = {
            "Hyper-SD": {"type": "hyper-sd", "steps": 1},  # 保持1步以保证速度
            "LCM": {"type": "lcm", "steps": 8},  # 增加到8步以提高质量
            "SD1.5-Base": {"type": "base", "steps": 25}  # 保持25步并提高质量参数
        }
        
        # 存储结果
        results = {}
        original_image = verify_image(input_image_path)
        original_image.save("TSCD/original.jpg")
        
        # 测试每个模型
        for model_name, config in model_configs.items():
            try:
                print(f"\n正在测试 {model_name}...")
                print(f"正在加载模型...")
                pipe = load_model(config["type"])
                
                print(f"开始推理...")
                output_image, time_used, psnr, ssim_score = test_pipeline(
                    pipe, 
                    input_image_path, 
                    model_type=config["type"],
                    steps=config["steps"]
                )
                
                # 保存结果
                output_image.save(f"TSCD/{model_name.lower()}_result.jpg")
                results[model_name] = {
                    "steps": config["steps"],
                    "time": time_used,
                    "psnr": psnr,
                    "ssim": ssim_score
                }
                
            except Exception as e:
                print(f"处理 {model_name} 时发生错误: {str(e)}")
                results[model_name] = {
                    "steps": config["steps"],
                    "time": 0,
                    "psnr": 0,
                    "ssim": 0
                }
            finally:
                # 清理 GPU 内存
                if 'pipe' in locals():
                    del pipe
                torch.cuda.empty_cache()
        
        # 打印结果表格
        print("\n测试结果:")
        print(f"{'模型':<12} {'步数':<6} {'时间(秒)':<10} {'PSNR':<8} {'SSIM':<8}")
        print("-" * 44)
        for model_name, metrics in results.items():
            print(f"{model_name:<12} {metrics['steps']:<6} {metrics['time']:.2f}s{' ':<6} {metrics['psnr']:.2f}{' ':<4} {metrics['ssim']:.4f}")
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise