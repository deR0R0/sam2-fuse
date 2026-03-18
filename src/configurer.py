import json

# singleton class
class Configurer:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Configurer, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)
    
    def get_pip_cmd(self) -> str:
        return self.config.get("pip_cmd", "pip")
    
    def get_python_path(self) -> str:
        return self.config.get("python_path", "python")
    
    def get_models(self) -> dict[str, str]:
        return self.config.get("models", {})