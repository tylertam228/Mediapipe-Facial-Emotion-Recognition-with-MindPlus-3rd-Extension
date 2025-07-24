<div align="center">
  <img src="python/_images/featured.png" alt="MediaPipe FER Extension Icon" width="200"/>
  
  # MediaPipe Facial Emotion Recognition Extension for Mind+
</div>

This project is a Mind+ extension for real-time facial emotion recognition using MediaPipe. It supports recognizing emotions like Happy, Sad, Angry, Surprised, and Neutral through rule-based facial landmark analysis.

## References and Acknowledgments

This project is based on and references the following works:
- [Facial-emotion-recognition-using-mediapipe](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe)
- [DFRobot Community Project](https://community.dfrobot.com/makelog-314255.html)

**Important Note**: This implementation uses rule-based data processing for emotion recognition and does not include any trained machine learning models. Therefore, the accuracy not 100% perfect.

## Features

- Real-time facial emotion recognition
- Support for multiple emotions: Happy, Sad, Angry, Surprised, Neutral
- Configurable camera settings and UI options
- Portrait and landscape display modes
- Face mesh visualization toggle
- Emotion counting and statistics
- Mind+ block-based programming interface

## Installation for Unihiker

Since Unihiker only comes with Python 3.7.3 which is not compatible with MediaPipe, you need to upgrade to Python 3.12 using pyenv:

### Prerequisites Installation

```bash
# Install dependencies
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to system configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Apply configuration
source ~/.bashrc

# Check pyenv version
pyenv --version
```

### Python 3.12 Installation

1. Download Python 3.12 version from: https://github.com/liliang9693/unihiker-pyenv-python/releases/download/3.8-3.13/python-3.12.7.tar.gz

2. Upload the tar.gz file to `/root/.pyenv/versions` via http://10.1.2.3/pc/file-upload (when Unihiker is connected)

3. Extract and setup:
```bash
# Navigate to the directory
cd /root/.pyenv/versions/

# Extract the downloaded file
tar -xzf python-3.12.7.tar.gz

# Refresh pyenv python list
pyenv rehash

# List available versions
pyenv versions

# Set 3.12.7 as global default python
pyenv global 3.12.7

# Verify python version
python --version

# Check pip package list
pip list
```

### Dependency Installation

```bash
# Upgrade pip
/usr/bin/python3 -m pip install --upgrade pip

# Install basic packages
pip install unihiker pinpong pandas
```

### MediaPipe and OpenCV Installation

Download the wheel files:
- OpenCV: https://files.pythonhosted.org/packages/f3/bd/29c126788da65c1fb2b5fb621b7fed0ed5f9122aa22a0868c5e2c15c6d23/opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
- MediaPipe: https://files.pythonhosted.org/packages/a8/f2/c8f62565abc93b9ac6a9936856d3c3c144c7f7896ef3d02bfbfad2ab6ee7/mediapipe-0.10.18-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

**Option 1**: Upload to Mind+ project and create dependency.py:
```python
import os
os.system("pip3 install opencv*.whl mediapipe*.whl")
```

**Option 2**: Upload to Unihiker directly and install:
```bash
pip3 install opencv*.whl mediapipe*.whl
```

## Usage in Mind+

1. Install this extension in Mind+
2. Use the blocks to set up emotion detection:
   - Initialize MediaPipe facial emotion recognition
   - Setup camera source and resolution
   - Configure UI options
   - Start emotion detection
3. Use detection blocks to check emotions and get confidence levels
4. Control the display with toggle face mesh and verbose mode options

## Block Categories

- **Basic Configuration**: Initialize, setup camera, set detection parameters, UI options
- **Emotion Detection**: Get current emotion, check specific emotions, emotion counting
- **Control**: Toggle face mesh display, set verbose mode

## Controls

- Press 'a' to exit the application
- Press 'b' to toggle face mesh display

## License

MIT License

---

# MediaPipe 臉部情緒辨識 Mind+ 擴展

這個專案是一個基於 MediaPipe 的即時臉部情緒辨識 Mind+ 擴展。它支援透過基於規則的臉部特徵點分析來辨識開心、難過、憤怒、驚訝和中性等情緒。

## 參考資料與致謝

本專案基於並參考了以下作品：
- [Facial-emotion-recognition-using-mediapipe](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe)
- [DFRobot 社群專案](https://community.dfrobot.com/makelog-314255.html)

**重要說明**：本實作使用基於規則的資料處理進行情緒辨識，未包含任何訓練過的機器學習模型。因此準確度不是 100%。

## 功能特色

- 即時臉部情緒辨識
- 支援多種情緒：開心、難過、憤怒、驚訝、中性
- 可配置的攝影機設定和 UI 選項
- 直向和橫向顯示模式
- 臉部網格視覺化切換
- 情緒計數和統計
- Mind+ 積木式程式設計介面

## Unihiker 安裝說明

由於 Unihiker 僅提供 Python 3.7.3，而 MediaPipe 不相容此版本，您需要使用 pyenv 升級到 Python 3.12：

### 安裝前置需求

```bash
# 安裝依賴庫
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

# 安裝 pyenv
curl https://pyenv.run | bash

# 將 pyenv 添加到系統配置中
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# 使配置生效
source ~/.bashrc

# 查看 pyenv 版本
pyenv --version
```

### Python 3.12 安裝

1. 從以下連結下載 Python 3.12 版本：https://github.com/liliang9693/unihiker-pyenv-python/releases/download/3.8-3.13/python-3.12.7.tar.gz

2. 透過 http://10.1.2.3/pc/file-upload 將 tar.gz 檔案上傳到 `/root/.pyenv/versions`（當 Unihiker 已連接時）

3. 解壓縮和設定：
```bash
# 進入目錄
cd /root/.pyenv/versions/

# 解壓縮下載的檔案
tar -xzf python-3.12.7.tar.gz

# 刷新 pyenv python 清單
pyenv rehash

# 列出可用版本
pyenv versions

# 設定 3.12.7 為全域預設 python
pyenv global 3.12.7

# 驗證 python 版本
python --version

# 檢查 pip 套件清單
pip list
```

### 依賴套件安裝

```bash
# 升級 pip
/usr/bin/python3 -m pip install --upgrade pip

# 安裝基本套件
pip install unihiker pinpong pandas
```

### MediaPipe 和 OpenCV 安裝

下載 wheel 檔案：
- OpenCV：https://files.pythonhosted.org/packages/f3/bd/29c126788da65c1fb2b5fb621b7fed0ed5f9122aa22a0868c5e2c15c6d23/opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
- MediaPipe：https://files.pythonhosted.org/packages/a8/f2/c8f62565abc93b9ac6a9936856d3c3c144c7f7896ef3d02bfbfad2ab6ee7/mediapipe-0.10.18-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

**選項一**：上傳到 Mind+ 專案並建立 dependency.py：
```python
import os
os.system("pip3 install opencv*.whl mediapipe*.whl")
```

**選項二**：直接上傳到 Unihiker 並安裝：
```bash
pip3 install opencv*.whl mediapipe*.whl
```

## 在 Mind+ 中的使用方法

1. 在 Mind+ 中安裝此擴展
2. 使用積木設定情緒偵測：
   - 初始化 MediaPipe 臉部情緒辨識
   - 設定攝影機來源和解析度
   - 配置 UI 選項
   - 開始情緒偵測
3. 使用偵測積木檢查情緒並取得信心度
4. 使用切換臉部網格和詳細模式選項控制顯示

## 積木分類

- **基本設定**：初始化、設定攝影機、偵測參數、UI 選項
- **情緒偵測**：取得目前情緒、檢查特定情緒、情緒計數
- **控制**：切換臉部網格顯示、設定詳細模式

## 控制按鍵

- 按 'a' 鍵退出應用程式
- 按 'b' 鍵切換臉部網格顯示

## 授權條款

MIT 授權條款

## Project Structure / 專案結構

```
Mind+ (Mediapipe Emotion)/
├── config.json                    # Extension configuration / 擴展配置
├── README.md                      # Project documentation / 專案文件
└── python/                       # Python source files / Python 原始檔案
    ├── main.ts                   # Block definitions / 積木定義
    ├── _images/                  # Extension icons / 擴展圖示
    │   ├── featured.png
    │   └── icon.svg
    ├── _locales/                 # Language files / 語言檔案
    │   ├── en.json               # English translations / 英文翻譯
    │   └── zh-cn.json            # Traditional Chinese translations / 繁體中文翻譯
    ├── _menus/                   # Menu definitions / 選單定義
    │   └── index.json
    └── libraries/                # Python libraries / Python 函式庫
        ├── mediapipe_FER.py      # Original FER implementation / 原始 FER 實作
        └── mediapipe_fer_blocks.py # Block interface library / 積木介面函式庫
```
