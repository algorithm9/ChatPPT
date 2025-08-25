# src/image_generator.py
import os
import torch
from diffusers import AutoPipelineForText2Image
from langchain_core.prompts import ChatPromptTemplate
from logger import LOG
from config import Config
from model_factory import get_model

class ImageGenerator:
    """
    使用本地 Stable Diffusion 模型为幻灯片内容生成图片。
    """
    def __init__(self, prompt_file):
        self.config = Config()
        self.prompt_template = self._load_prompt_template(prompt_file)
        self.llm = get_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOG.info(f"Using device for image generation: {self.device}")
        if self.device == "cpu":
            LOG.warning("Running Stable Diffusion on CPU. This will be very slow.")
        
        self.pipeline = self._init_sd_pipeline()

    def _load_prompt_template(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            LOG.error(f"Prompt file not found: {file_path}")
            raise

    def _init_sd_pipeline(self):
        """
        初始化 Stable Diffusion pipeline。
        """
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.config.sd_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
            return pipe.to(self.device)
        except Exception as e:
            LOG.error(f"Failed to load Stable Diffusion model: {e}")
            LOG.error("Please ensure you have a GPU with sufficient VRAM and CUDA is correctly installed.")
            raise

    def _generate_image_prompt(self, slide_content):
        """
        使用 LLM 将幻灯片内容转换为文生图模型的提示词。
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            ("human", slide_content),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"input": slide_content})
        image_prompt = response.content.strip()
        LOG.debug(f"Generated Image Prompt: {image_prompt}")
        return image_prompt

    def generate_image(self, slide_title, slide_content_text):
        """
        生成图片并返回其本地保存路径。
        """
        try:
            image_prompt = self._generate_image_prompt(slide_content_text)
            
            # 使用本地 pipeline 生成图片
            # 添加一些通用的高质量后缀以提升图片效果
            full_prompt = f"{image_prompt}, 8k, photorealistic, cinematic lighting"
            image = self.pipeline(prompt=full_prompt).images[0]
            
            # 保存图片
            save_dir = "images/generated"
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"{slide_title.replace(' ', '_')}.png"
            save_path = os.path.join(save_dir, file_name)
            
            image.save(save_path)
            
            LOG.info(f"Image saved to {save_path}")
            return save_path

        except Exception as e:
            LOG.error(f"Failed to generate or save image for slide '{slide_title}': {e}")
            return None