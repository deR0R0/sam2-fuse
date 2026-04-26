# SAM2-Fuse

A Davinci Resolve "plugin" (fuse) that allows for faster workflows by masking easily with just a few points!

> [!WARNING]
> SAM2-Fuse is currently in beta and only supports macOS (my daily driver). Windows will come as soon as it's out of beta.

> [!IMPORTANT]
> SAM2-Fuse may be a little bit "unstable" at times. Simply restart the backend server or Davinci Resolve to fix the issues.

# 🚀 Features
- Easy setup (just run a single script)
- Add foreground and background points for fine control
- Support for CUDA GPUs
- Hands-free after you start tracking! Work is sent off to the local server for processing.

# 🎬 Demo

https://github.com/user-attachments/assets/835c9805-ff14-4e19-9014-e3ba0e7ca733

# ⬇️ Installation

1. Go to releases and download the latest version according to your operating system
2. Ensure python >=3.12 is downloaded on your machine. If not: 
- Windows: Go to https://python.org and download.
- macOS: Downloading using homebrew: `brew install python`
3. Run `setup.py` by double clicking on windows or on mac: `python3 setup.py`. It'll prompt you for installation: choose first option and for models, choose whatever you prefer.
4. **IMPORTANT STEP** Drag the sam2-fuse folder to the fuses folder on your operating system:
- Windows: `C:\Users\[Username]\AppData\Roaming\Blackmagic Design\DaVinci Resolve\Support\Fusion\Fuses`
- macOS: `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Fuses`
5. You can use it now :)

# 🐞 Bugs? Report them in issues or fix it via pull request!

# 📈 High RAM Usage?

It's due to SAM2's high memory footprint. I've tried adding things that'll help with the issue but if it's too much, contact me `me@robertzhao.dev` if you want a **paid** cloud solution.
