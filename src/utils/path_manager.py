from pathlib import Path
import yaml


class PathManager:
    def __init__(self, config_file: Path = None):
        if config_file is None:
            self._script_dir = Path(__file__).parent
            self._project_root = self._script_dir.parent.parent
            config_file = self._project_root / "config" / "paths.yml"
        else:
            config_file = Path(config_file).resolve()
            self._project_root = config_file.parent

        self._config = self._load_config(config_file)

        project_root_cfg = self._config.get("project_root", ".")
        if Path(project_root_cfg).is_absolute():
            self._project_root = Path(project_root_cfg).resolve()
        else:
            self._project_root = (config_file.parent.parent / project_root_cfg).resolve()

        self._init_paths()

    def _load_config(self, config_file: Path) -> dict:
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件{config_file}不存在")
        with config_file.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _init_paths(self) -> None:
        self.PROJECT_ROOT = self._project_root

        # 数据集目录
        data_cfg = self._config.get("data_dir", {})
        self.WESAD_ROOT = self._project_root / data_cfg.get("name", "WESAD")

        # 主目录
        src_cfg = self._config.get("src_dir", {})
        self.SRC_ROOT = self._project_root / src_cfg.get("name", "src")

def get_path_manager(config_file: Path = None) -> PathManager:
    return PathManager(config_file)
