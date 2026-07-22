# Persistent Context & Rules: Mini Golf AI Tracking System

This file serves as a persistent reference for future agent sessions working on this repository.

---

## 1. Camera Hardware & Configuration
* **Sensor:** Raspberry Pi Camera Module 3 (Sony IMX708) with 16:9 aspect ratio.
* **Optimal Sensor Settings (Full FOV, No Cropping):**
  ```bash
  rpicam-vid -t 0 --mode 2304:1296:10 --width 960 --height 540 --framerate 30 --inline --codec h264 --profile baseline --listen -o tcp://0.0.0.0:5000
  ```
  *(Note: Mode `2304:1296:10` or wide-angle mode gives the full wide FOV of the IMX708 sensor, which is essential to capture the entire course).*

---

## 2. Model & Class Mapping Rules
* Standard COCO models (like `yolo26n.pt`) use class `32` for a sports ball.
* Custom single-class models (like `best_combined_yolo26.pt`) use class `0` for the `golf-ball`.
* **Important Detection Rule:** The detector code must automatically inspect `len(model.names)` at startup. If `len(model.names) == 1`, override `class_id` to `0`. If it's a standard COCO model, map the target to `32`.
* **Confidence Thresholds:** For the custom model, set the threshold to **`0.70`** in `config/config.yaml` to prevent shoes, putter heads, and player shadows from registering as false positives.

---

## 3. Dataset Structures
* **Orange & White Dataset:** Stored in `datasets/golf_dataset.zip` (1,427 frames).
* **Multi-Color Dataset (Red, Yellow, Green, Orange, White):** Stored in `multicolor_dataset/` as two separate split archives to stay under the 100MB Roboflow upload limit:
  * 📦 `multicolor_dataset/raw_images_part1.zip` (59.5 MB)
  * 📦 `multicolor_dataset/raw_images_part2.zip` (60.8 MB)

---

## 4. Key Performance Lessons
* **Socket Reconnect CPU Deadlocks:** In `RtspStream`, calling socket reads on an offline stream in a tight loop blocks the GIL and spikes CPU to 100%. **Always sleep for at least 2.0 seconds** on socket reconnect failures.
* **State Machine Speed Smoothing:** Fluctuations in sub-pixel bounding boxes can lock a resting ball in the `MOVING` state. Keep a 5-frame moving average deque for speed smoothing.
* **Laptop Hysteresis Thresholds:** For low-FPS developer testing, raise the stopped threshold to `0.20 m/s` and the moving threshold to `0.35 m/s` to prevent the state machine from being trapped in limbo.

---

## 5. Real-Time Color Identification & Stability
* **Heuristic HSV Classifier:** Cropped ball regions are converted to HSV. Green background (turf) pixels (`35 <= Hue <= 85`) are ignored. Remaining pixels classify color:
  * **White:** Saturation < 55 and Value > 130.
  * **Red:** Hue < 10 or Hue > 165.
  * **Orange:** 10 <= Hue < 25.
  * **Yellow:** 25 <= Hue < 38.
  * **Green:** 38 <= Hue < 85.
* **Color Stability (Majority Voting):** To prevent transient noise (e.g. putter shadows, reflections inside the cup) from flipping the ball's color, the tracker maintains a vote history (`color_votes`) per track and locks onto the majority color.
