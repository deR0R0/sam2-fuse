"""
setup.py
=========
Author: deR0R0
Description: This script is responsible for installing the necessary dependencies for SAM2-Fuse, including PyTorch. It detects the user's platform and installs the appropriate version of PyTorch based on the available hardware (e.g., CUDA for NVIDIA GPUs, MPS for Apple Silicon). Create virtual environment and install dependencies in it. It also handles error cases and provides informative logging throughout the installation process.
License: MIT License
"""

from datetime import datetime
from enum import Enum
import subprocess, os, sys, venv, shutil

class ExitCodes(Enum):
    NOTHING = 0
    SUCCESS = 1
    UNSUPPORTED_PLATFORM = 2
    INSTALLATION_FAILED = 3
    UNKNOWN_ERROR = 4
    RESTART_REQUIRED = 5

class Setup:
    def __init__(self):
        self.platform: str = self.get_platform().lower()
        self.pip_cmd: list[str] = ["python3", "-m", "pip"]
        self.python_path: str = sys.executable
        self.exit_codes: ExitCodes = ExitCodes.NOTHING
        self.exit_reason: str = ""
        self.models_installed: list[str] = []

    def get_time(self):
        return datetime.now().strftime("%H:%M:%S")

    def log(self, message):
        print(f"[{self.get_time()}] {message}")

    def error(self, message):
        print(f"[{self.get_time()}] ERROR: {message}")

    def prompt(self, message, options: dict[str, str]):
        self.log("======== PROMPT ========")
        self.log(message)
        for key, value in options.items():
            print(f"{key}: {value}")
        choice = input("Enter your choice: ")

        if choice not in options.keys() and choice != "":
            self.log("Invalid choice. Please try again.")
            return self.prompt(message, options)

        return choice if choice != "" else "1"

    def get_platform(self):
        import platform
        return platform.system()
    
    def install_packages(self, packages: list[str]):
        for package in packages:
            self.log(f"Installing package: {package}")
            try:
                subprocess.check_call(self.pip_cmd + [package])
            except subprocess.CalledProcessError as e:
                self.error(f"Failed to install package '{package}': {e}")
                self.exit_codes = ExitCodes.INSTALLATION_FAILED
                self.exit_reason = f"Failed to install package '{package}'."
                return
            
    def create_venv(self, name="sam2-fuse-env"):
        self.log(f"Creating virtual environment '{name}'...")
        
        # create virtual environment
        if not os.path.exists(name):
            venv.create(name, with_pip=True)
            self.log(f"Virtual environment '{name}' created successfully.")
        else:
            self.log(f"Virtual environment '{name}' already exists. Skipping creation.")
            
        # set python path
        self.python_path = os.path.join(name, "Scripts", "python.exe") if self.platform == "windows" else os.path.join(name, "bin", "python")
        self.pip_cmd = [self.python_path, "-m", "pip"]
        self.log(f"Using Python executable at: {self.python_path}")

        # install get-pip.py in the virtual environment to ensure pip is available
        try:
            subprocess.check_call(["curl", "-sS", "https://bootstrap.pypa.io/get-pip.py", "-o", "get-pip.py"])
            subprocess.check_call([self.python_path, "get-pip.py"])
            self.pip_cmd = ["pip".join(self.python_path.rsplit("python", 1)), "install"]
        except subprocess.CalledProcessError as e:
            self.error(f"Failed to install get-pip.py: {e}")
            self.exit_codes = ExitCodes.INSTALLATION_FAILED
            self.exit_reason = "Failed to install get-pip.py."
            return
        
        # delte get-pip.py after installation
        get_pip_path = os.path.abspath(os.path.join(os.getcwd(), "get-pip.py")) # use abs path to avoid issues with relative paths
        if os.path.exists(get_pip_path):
            os.remove(get_pip_path)
            self.log("Cleaned up get-pip.py after installation.")
        else:
            self.log("get-pip.py not found for cleanup. Skipping.")


    def confirm_cuda_support(self, version) -> str | None:
        self.log(f"Confirming CUDA actually exists for PyTorch...")
        pytorch_index_url = f"https://download.pytorch.org/whl/cu{version.replace('.', '')}"

        # send a HEAD request to the index url
        # 403 = basically doesn't exist
        # 404 = definitely doesn't exist
        # 200 = exists
        try:
            result = subprocess.check_output(["curl", "-I", pytorch_index_url], stderr=subprocess.STDOUT, text=True)
            if "200 OK" in result:
                self.log(f"Confirmed CUDA version {version} exists for PyTorch.")
                return version
            else:
                self.log(f"CUDA version {version} does not seem to exist for PyTorch. Response: {result}")
                return None
        except subprocess.CalledProcessError as e:
            self.error(f"Failed to confirm CUDA version with curl: {e}")
            return None
            

    def check_cuda_version(self):
        self.log("Attempting to check CUDA version with nvcc...")

        try:
            result = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
            for line in result.splitlines():
                if "release" in line:
                    version = line.split("release")[-1].split(",")[0].strip()
                    self.log(f"Detected CUDA version: {version}")
                    return version
            self.log("Could not determine CUDA version from nvcc output.")
        except subprocess.CalledProcessError as e:
            self.log(f"nvcc not found or failed to execute: {e}")

        # if the nvcc version check fails, we can try using
        # nvidia-smi as a fallback.
        # although, this returns the MAX CUDA version supported by the driver, not necessarily the version installed on the system
        self.log("Attempting to check CUDA version with nvidia-smi...")
        try:
            result = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
            for line in result.splitlines():
                if "CUDA Version" in line:
                    version = line.split("CUDA Version:")[-1].strip()
                    self.log(f"Detected CUDA version from nvidia-smi: {version}")
                    return version
            self.log("Could not determine CUDA version from nvidia-smi output.")
        except subprocess.CalledProcessError as e:
            self.log(f"nvidia-smi not found or failed to execute: {e}")
            
        return None
    
    def install_pytorch_generic(self):
        self.log("Installing Generic PyTorch with command: " + " ".join(self.pip_cmd + ["torch", "torchvision"]))
        try:
            subprocess.check_call(self.pip_cmd + ["torch", "torchvision"])
        except subprocess.CalledProcessError as e:
            self.error(f"Failed to install PyTorch: {e}")
            self.exit_codes = ExitCodes.INSTALLATION_FAILED
            self.exit_reason = "Failed to install PyTorch."
            return
        
    def install_pytorch_cuda(self):
        self.log("Installing PyTorch for CUDA...")

        # get cuda version
        cuda_version = self.check_cuda_version()
        if not cuda_version:
            self.log("COULD NOT DETERMINE CUDA VERSION. INSTALLING GENERIC PYTORCH.")
            return self.install_pytorch_generic()
        
        # map cuda version to pytorch version
        pytorch_index_url = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
        self.log(f"Using PyTorch index URL: {pytorch_index_url}")

        # install pytorch with the appropriate index url
        try:
            subprocess.check_call(self.pip_cmd + ["torch", "torchvision", "--index-url", pytorch_index_url])
        except subprocess.CalledProcessError as e:
            self.error(f"Failed to install PyTorch with CUDA support: {e}")
            self.exit_codes = ExitCodes.INSTALLATION_FAILED
            self.exit_reason = "Failed to install PyTorch with CUDA support."
            return
        
    def install_sam2_models(self, models: list[str]):
        self.log(f"Installing SAM2 models: {', '.join(models)}")

        url_base: str = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
        
        model_urls = {
            "large": "sam2.1_hiera_large.pt",
            "base": "sam2.1_hiera_base_plus.pt",
            "small": "sam2.1_hiera_small.pt",
            "tiny": "sam2.1_hiera_tiny.pt",
        }

        # install all the models to a "models" directory
        if not os.path.exists("models"):
            os.makedirs("models")

        # delete everything in that directory first to avoid confusion with old models
        for filename in os.listdir("models"):
            file_path = os.path.abspath(os.path.join("models", filename))
            if os.path.isfile(file_path):
                os.remove(file_path)
                self.log(f"Removed old model file: {file_path}")

        for model in models:
            if model in model_urls:
                url = model_urls[model]
                self.log(f"Installing SAM2 {model} model from {url}...")
                try:
                    subprocess.check_call(["curl", "-L", f"{url_base}{url}", "-o", os.path.abspath(os.path.join("./models", f"{model_urls[model]}"))])
                    self.models_installed.append(model_urls[model])
                except subprocess.CalledProcessError as e:
                    self.error(f"Failed to install SAM2 {model} model: {e}")
                    self.exit_codes = ExitCodes.INSTALLATION_FAILED
                    self.exit_reason = f"Failed to install SAM2 {model} model."
                    return
            else:
                self.log(f"Unknown model '{model}' specified. Skipping.")


    def install_sam2(self):
        self.log("Cloning SAM2 repo...")
        
        # download github repo with pip
        self.install_packages(["https://github.com/facebookresearch/segment-anything-2/archive/refs/heads/main.zip"])

        if self.exit_codes != ExitCodes.NOTHING:
            self.error("Aborting SAM2 installation due to previous package installation failure.")
            self.error("ERROR: " + self.exit_reason)
            return
        
        # prompt for which models the user wants to install
        options = {
            "1": "Install all SAM2 models ... 1.48 GiB ... (default, suggested)",
            "2": "Install only large ... 846 MiB ...",
            "3": "Install only base ... 309 MiB ...",
            "4": "Install only small ... 176 MiB ...",
            "5": "Install only tiny ... 149 MiB ...",
        }

        choice = self.prompt("What SAM2 models would you like to install?", options)
        self.log(f"======== USER CHOICE: {choice} =======")

        match choice:
            case "1":
                self.install_sam2_models(["large", "base", "small", "tiny"])
            case "2":
                self.install_sam2_models(["large"])
            case "3":
                self.install_sam2_models(["base"])
            case "4":
                self.install_sam2_models(["small"])
            case "5":
                self.install_sam2_models(["tiny"])
            case _:
                self.log("Invalid choice. Installing all models by default.")
                self.install_sam2_models(["large", "base", "small", "tiny"])


    def install_lua_json(self, platform: str):
        folder = ""
        lua_file = "json.lua"

        if platform.lower() == "darwin":
            folder = os.path.expanduser("~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Modules/Lua/json.lua")
        elif platform.lower() == "windows":
            folder = os.path.join(os.environ["PROGRAMDATA"], "Blackmagic Design", "DaVinci Resolve", "Fusion", "Modules", "Lua", "json.lua")

        if folder == "":
            self.error("There was a problem finding the path of the davinci resolve fusion modules folder. Please install 'json.lua' manually.")
            self.exit_codes = ExitCodes.INSTALLATION_FAILED
            self.exit_reason = "Problem finding path of davinci resolve fusion modules folder."
            return
        
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        shutil.copy(lua_file, folder)

    
    def install(self):
        """
        NVIDIA CUDA: determine the CUDA version and download the correct type of pytorch
        MPS/OTHERS: just install the standard pytorch version

        We're going to have to make virtual environments now...
        """
        self.log(f"Starting SAM2, Pytorch installation for {self.platform} platform.")

        if self.exit_codes != ExitCodes.NOTHING:
            self.error("Aborting installation due to pip installation failure.")
            return

        # check if the platform is supported
        if self.platform == "linux":
            self.error("Linux is not supported yet. Please use Windows or Mac.")
            self.error("You may create a pull request to add Linux support @ https://github.com/deR0R0/sam2-fuse!")
            self.exit_codes = ExitCodes.UNSUPPORTED_PLATFORM
            self.exit_reason = "Linux is currently unsupported."
            return
        
        # create virtual environment
        self.create_venv()

        if self.exit_codes != ExitCodes.NOTHING:
            self.error("Aborting installation due to virtual environment creation failure.")
            return
        
        # check for macOS
        if self.platform == "darwin":
            self.log("Detected macOS platform. Installing PyTorch")

            # install pytorch generic (include mps)
            self.install_pytorch_generic()

            # mac cannot install decord, so we will just use the bundled version that i hand compiled ;(
            # pip install . inside of the decord folder
            self.log("Installing decord for macOS from bundled version...")
            self.install_packages(["./decord/python"])

        # check for windows
        if self.platform == "windows":
            self.log("Detected Windows platform. Installing PyTorch with CUDA support if available...")
            self.install_pytorch_cuda()
            self.install_packages(["decord"]) # on windows, decord can be installed normally

        self.install_packages(["fastapi", "pillow"])

        # install SAM2 models
        self.install_sam2()

        if self.exit_codes == ExitCodes.NOTHING:
            self.exit_codes = ExitCodes.SUCCESS
            self.exit_reason = "Python dependencies installed successfully."


        # install lua sht. i hate this
        self.install_lua_json(self.platform)

    def install_models(self):
        self.log("Installing SAM2 models...")
        self.install_sam2()


if __name__ == "__main__":
    setup = Setup()

    options = {
        "1": "Install SAM2-Fuse dependencies (PyTorch, SAM2 models, etc.) (suggested)",
        "2": "Install only SAM2 models (if you followed the instructions in the README)",
    }

    result = setup.prompt("Hey! This is the SAM2-Fuse installer for davinci resolve. Please read the README or watch the video guide for installation instructions. What would you like to do?", options)

    if result == "1":
        setup.install()
    else:
        setup.install_models()

    if setup.exit_codes != ExitCodes.NOTHING and setup.exit_codes != ExitCodes.SUCCESS:
        setup.error(f"Installation failed with exit code {setup.exit_codes.value}: {setup.exit_reason}")
    else:
        setup.log("Installation completed successfully!")
        models_dir = {}
        for model in setup.models_installed:
            models_dir[model] = os.path.abspath(os.path.join("./models", model))
        config = {
            "pip_cmd": setup.pip_cmd,
            "python_path": setup.python_path,
            "models": models_dir
        }
        with open("config.json", "w") as f:
            import json
            json.dump(config, f, indent=4)