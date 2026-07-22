import cv2
import os

def main():
    video_path = "D:\\AI Projects\\Golf AI\\Training Videos\\input3.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video resolution: {w}x{h}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    
    # Save a few sample frames to inspect
    for frame_no in [100, 300, 600]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret and frame is not None:
            filename = f"sample_frame_{frame_no}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            
    cap.release()

if __name__ == "__main__":
    main()
