# Google Colab Setup & Run Guide

Copy and paste the code block below into a single cell in your Google Colab notebook. Ensure your runtime hardware accelerator is set to **T4 GPU** before execution.

```python
# 1. Mount Google Drive
from google.colab import drive
import os
drive.mount('/content/drive')

# 2. Clone repository if it doesn't exist
if not os.path.exists('/content/GolfAI'):
    !git clone https://github.com/mayurrupala47/GolfAI.git /content/GolfAI

# Permanently switch the kernel directory to the repository
%cd /content/GolfAI

# Reset working directory to the absolute latest version on GitHub
!git fetch origin main
!git reset --hard origin/main

# 3. Copy the TrackNet model weights from your Google Drive
os.makedirs('/content/GolfAI/models', exist_ok=True)
# NOTE: Update the source paths below if your models or videos are in different Drive folders
!cp "/content/drive/MyDrive/Golf AI/models/TrackNet_best.pt" "/content/GolfAI/models/TrackNet_best.pt"
!cp "/content/drive/MyDrive/Golf AI/Golf All color videos/orange_right_1.mp4" "/content/GolfAI/orange_right_1.mp4"

# 4. Install dependencies
!pip install -q -r /content/GolfAI/requirements.txt

# 5. Run the tracking pipeline using the PyTorch model with color identification enabled
!python /content/GolfAI/main.py --video "/content/GolfAI/orange_right_1.mp4" --detector tracknet
```
