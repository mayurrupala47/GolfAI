import cv2
import os
import numpy as np

frames_dir = r"scratch/stroke6_frames"
files = sorted(os.listdir(frames_dir))

for f in files:
    if not f.endswith(".png"):
        continue
    frame_no = f.split("_")[1].split(".")[0]
    img_path = os.path.join(frames_dir, f)
    img = cv2.imread(img_path)
    
    # Crop 1: Around rest position (244.8, 319.3) -> x:[194, 294], y:[269, 369]
    crop_rest = img[269:369, 194:294]
    
    # Crop 2: Around shoe/stroke6 position (290.5, 241.2) -> x:[240, 340], y:[191, 291]
    crop_shoe = img[191:291, 240:340]
    
    # Measure orange pixels in both crops (HSV 5-28, S>70, V>70)
    hsv_rest = cv2.cvtColor(crop_rest, cv2.COLOR_BGR2HSV)
    hsv_shoe = cv2.cvtColor(crop_shoe, cv2.COLOR_BGR2HSV)
    
    mask_rest = (hsv_rest[:,:,0] >= 5) & (hsv_rest[:,:,0] < 28) & (hsv_rest[:,:,1] > 70) & (hsv_rest[:,:,2] > 70)
    mask_shoe = (hsv_shoe[:,:,0] >= 5) & (hsv_shoe[:,:,0] < 28) & (hsv_shoe[:,:,1] > 70) & (hsv_shoe[:,:,2] > 70)
    
    cnt_rest = np.sum(mask_rest)
    cnt_shoe = np.sum(mask_shoe)
    
    print(f"Frame {frame_no}: Rest_Orange_Pixels={cnt_rest:3d} | Shoe_Orange_Pixels={cnt_shoe:3d}")
