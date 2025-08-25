# src/model_factory.py
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from config import Config
import os

def get_model():
    """
    根据配置动态创建并返回一个大语言模型实例。
    """
    config = Config()
    if config.llm_provider == "deepseek":
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DeepSeek API key not found. Please set it in config.json or as an environment variable DEEPSEEK_API_KEY.")
        return ChatDeepSeek(
            model=config.deepseek_model,
            temperature=0.5,
            max_tokens=4096,
        )
    else: # 默认为 openai
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=4096,
        )