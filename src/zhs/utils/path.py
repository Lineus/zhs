"""路径工具模块"""

from pathlib import Path


def get_data_dir() -> Path:
    """获取数据目录（~/.zhs/，与当前工作目录无关），不存在时自动创建

    使用固定的用户级目录，保证全局安装的 ``zhs`` 命令在任何目录下运行
    都读写同一份配置、Cookie 与缓存。
    """
    data_dir = Path.home() / ".zhs"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_config_path() -> Path:
    """获取配置文件路径（~/.zhs/config.toml）"""
    return get_data_dir() / "config.toml"
