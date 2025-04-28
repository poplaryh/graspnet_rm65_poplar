import yaml
import os
from typing import Dict

def load_config(config_path: str = "config/para.yaml") -> Dict:
    """加载 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"配置文件 {config_path} 未找到")
    except yaml.YAMLError as e:
        raise Exception(f"YAML 解析错误: {str(e)}")