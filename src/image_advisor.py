import re
import requests
import os

from abc import ABC
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from logger import LOG  # 导入日志工具
from model_factory import get_model

from config import Config
from image_generator import ImageGenerator 

class ImageAdvisor(ABC):
    """
    聊天机器人基类，提供建议配图的功能。
    """
    def __init__(self, prompt_file="./prompts/image_advisor.txt"):
        self.config = Config()
        self.prompt_file = prompt_file
        self.prompt = self.load_prompt()
        self.create_advisor()
        if self.config.image_provider == 'stable_diffusion':
            self.image_generator = ImageGenerator(self.config.image_generator_prompt)


    def load_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            LOG.error(f"找不到提示文件 {self.prompt_file}!")
            raise

    def create_advisor(self):
        """
        初始化聊天机器人，包括系统提示和消息历史记录。
        """
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),  # 系统提示部分
            ("human", "**Content**:\n\n{input}"),  # 消息占位符
        ])

        self.model = get_model()
        self.advisor = chat_prompt | self.model

    def generate_images(self, markdown_content, image_directory="tmps", num_images=3):
        """
        生成图片并嵌入到指定的 PowerPoint 内容中。

        参数:
            markdown_content (str): PowerPoint markdown 原始格式
            image_directory (str): 本地保存图片的文件夹名称
            num_images (int): 每个幻灯片搜索的图像数量

        返回:
            content_with_images (str): 嵌入图片后的内容
            image_pair (dict): 每个幻灯片标题对应的图像路径
        """
        response = self.advisor.invoke({"input": markdown_content})
        LOG.debug(f"[Advisor 建议配图]\n{response.content}")
        
        # 将 Markdown 文本按幻灯片分割
        slides = self._split_markdown_by_slides(markdown_content)
        slide_contents = {title: content for title, content in slides}

        keywords = self.get_keywords(response.content)
        image_pair = {}

        for slide_title, query in keywords.items():
            save_path = None
            if self.config.image_provider == 'stable_diffusion':
                # 使用文生图模型
                slide_text = slide_contents.get(slide_title, "")
                if slide_text:
                    save_path = self.image_generator.generate_image(slide_title, slide_text)
            else:
                # 默认使用 Bing 搜索引擎
                images = self.get_bing_images(slide_title, query, num_images, timeout=1, retries=3)
                if images:
                    img = images[0]
                    save_directory = f"images/{image_directory}"
                    os.makedirs(save_directory, exist_ok=True)
                    save_path = os.path.join(save_directory, f"{img['slide_title']}_1.jpeg")
                    self.save_image(img["obj"], save_path)
            
            if save_path:
                image_pair[slide_title] = save_path

        content_with_images = self.insert_images(markdown_content, image_pair)
        return content_with_images, image_pair
    
    def _split_markdown_by_slides(self, markdown_content):
        """
        将完整的 markdown 文本分割成 (标题, 内容) 的元组列表。
        """
        slides = []
        lines = markdown_content.strip().split('\n')
        current_title = ""
        current_content = []

        for line in lines:
            if line.startswith('## '):
                if current_title:
                    slides.append((current_title, '\n'.join(current_content)))
                current_title = line[3:].strip()
                current_content = [line]
            elif current_title:
                current_content.append(line)
        
        if current_title:
            slides.append((current_title, '\n'.join(current_content)))
        
        return slides

    def get_keywords(self, advice):
        """
        使用正则表达式提取关键词。

        参数:
            advice (str): 提示文本
        返回:
            keywords (dict): 提取的关键词字典
        """
        pairs = re.findall(r'\[(.+?)\]:\s*(.+)', advice)
        keywords = {key.strip(): value.strip() for key, value in pairs}
        LOG.debug(f"[检索关键词 正则提取结果]{keywords}")
        return keywords

    def get_bing_images(self, slide_title, query, num_images=5, timeout=1, retries=3):
        """
        从 Bing 检索图像，最多重试3次。

        参数:
            slide_title (str): 幻灯片标题
            query (str): 图像搜索关键词
            num_images (int): 搜索的图像数量
            timeout (int): 每次请求超时时间（秒），默认1秒
            retries (int): 最大重试次数，默认3次

        返回:
            sorted_images (list): 符合条件的图像数据列表
        """
        url = f"https://www.bing.com/images/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
        }

        # 尝试请求并设置重试逻辑
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                break  # 请求成功，跳出重试循环
            except requests.RequestException as e:
                LOG.warning(f"Attempt {attempt + 1}/{retries} failed for query '{query}': {e}")
                if attempt == retries - 1:
                    LOG.error(f"Max retries reached for query '{query}'.")
                    return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        image_elements = soup.select("a.iusc")

        image_links = []
        for img in image_elements:
            m_data = img.get("m")
            if m_data:
                m_json = eval(m_data)
                if "murl" in m_json:
                    image_links.append(m_json["murl"])
            if len(image_links) >= num_images:
                break

        image_data = []
        for link in image_links:
            for attempt in range(retries):
                try:
                    img_data = requests.get(link, headers=headers, timeout=timeout)
                    img = Image.open(BytesIO(img_data.content))
                    image_info = {
                        "slide_title": slide_title,
                        "query": query,
                        "width": img.width,
                        "height": img.height,
                        "resolution": img.width * img.height,
                        "obj": img,
                    }
                    image_data.append(image_info)
                    break  # 成功下载图像，跳出重试循环
                except Exception as e:
                    LOG.warning(f"Attempt {attempt + 1}/{retries} failed for image '{link}': {e}")
                    if attempt == retries - 1:
                        LOG.error(f"Max retries reached for image '{link}'. Skipping.")
        
        sorted_images = sorted(image_data, key=lambda x: x["resolution"], reverse=True)
        return sorted_images

    def save_image(self, img, save_path, format="JPEG", quality=85, max_size=1080):
        """
        保存图像到本地并压缩。

        参数:
            img (Image): 图像对象
            save_path (str): 保存路径
            format (str): 保存格式，默认 JPEG
            quality (int): 图像质量，默认 85
            max_size (int): 最大边长，默认 1080
        """
        try:
            width, height = img.size
            if max(width, height) > max_size:
                scaling_factor = max_size / max(width, height)
                new_width = int(width * scaling_factor)
                new_height = int(height * scaling_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if img.mode == "RGBA":
                format = "PNG"
                save_options = {"optimize": True}
            else:
                save_options = {
                    "quality": quality,
                    "optimize": True,
                    "progressive": True
                }

            img.save(save_path, format=format, **save_options)
            LOG.debug(f"Image saved as {save_path} in {format} format with quality {quality}.")
        except Exception as e:
            LOG.error(f"Failed to save image: {e}")

    def insert_images(self, markdown_content, image_pair):
        """
        将图像嵌入到 Markdown 内容中。

        参数:
            markdown_content (str): Markdown 内容
            image_pair (dict): 幻灯片标题到图像路径的映射

        返回:
            new_content (str): 嵌入图像后的内容
        """
        lines = markdown_content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            new_lines.append(line)
            if line.startswith('## '):
                slide_title = line[3:].strip()
                if slide_title in image_pair:
                    image_path = image_pair[slide_title]
                    image_markdown = f'![{slide_title}]({image_path})'
                    new_lines.append(image_markdown)
            i += 1
        new_content = '\n'.join(new_lines)
        return new_content
