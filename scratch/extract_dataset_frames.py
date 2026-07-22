import cv2
import os
import shutil

def extract_frames():
    base_dir = "c:/Users/sspl260/.gemini/antigravity/scratch/mini-golf-ai"
    output_dir = os.path.join(base_dir, "multicolor_dataset/raw_images")
    
    # Recreate the output folder to ensure it is clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # List of videos to extract from
    videos = [
        {"name": "red", "file": "red_color_ball.mp4"},
        {"name": "yellow", "file": "yellow_color_ball.mp4"},
        {"name": "green", "file": "green_color_ball.mp4"},
        {"name": "orange", "file": "orange_color_ball.mp4"},
        {"name": "white", "file": "white_color_ball.mp4"}
    ]
    
    total_extracted = 0
    target_frames_per_video = 500

    print("Starting frame extraction...")
    
    for vid_info in videos:
        vid_path = os.path.join(base_dir, vid_info["file"])
        if not os.path.exists(vid_path):
            print(f"Warning: Video file {vid_path} not found! Skipping...")
            continue
            
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: Could not read frames from {vid_info['file']}!")
            cap.release()
            continue
            
        # Calculate step size to get exactly 500 frames evenly spaced
        step = max(1, total_frames // target_frames_per_video)
        print(f"Processing '{vid_info['file']}': Total frames={total_frames}, Extracting every {step}th frame...")
        
        count = 0
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0 and count < target_frames_per_video:
                img_name = f"{vid_info['name']}_frame_{frame_idx:06d}.jpg"
                img_path = os.path.join(output_dir, img_name)
                # Save frame in medium JPEG quality (75) to keep zip size well under 100MB
                cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                count += 1
                total_extracted += 1
                
            frame_idx += 1
            
        cap.release()
        print(f"Extracted {count} frames from {vid_info['file']}.")
        
    print(f"\nAll extractions complete! Total images in dataset: {total_extracted}")
    
    # Split the extracted images into two parts for zipping (to stay under 100MB limit)
    part1_dir = os.path.join(base_dir, "multicolor_dataset/raw_images_part1")
    part2_dir = os.path.join(base_dir, "multicolor_dataset/raw_images_part2")
    
    # Recreate clean part folders
    for d in [part1_dir, part2_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        
    all_files = sorted(os.listdir(output_dir))
    split_point = len(all_files) // 2
    
    print(f"Splitting 2500 images: {split_point} into Part 1, {len(all_files) - split_point} into Part 2...")
    
    for idx, f in enumerate(all_files):
        src_path = os.path.join(output_dir, f)
        if idx < split_point:
            shutil.move(src_path, os.path.join(part1_dir, f))
        else:
            shutil.move(src_path, os.path.join(part2_dir, f))
            
    # Zip both folders
    print("Compressing Part 1 into ZIP archive...")
    shutil.make_archive(part1_dir, 'zip', part1_dir)
    print("Compressing Part 2 into ZIP archive...")
    shutil.make_archive(part2_dir, 'zip', part2_dir)
    
    # Clean up folders to save disk space
    shutil.rmtree(output_dir)
    shutil.rmtree(part1_dir)
    shutil.rmtree(part2_dir)
    
    print("\nCompression finished! Created two ZIP files under 100MB:")
    print(f"1. multicolor_dataset/raw_images_part1.zip")
    print(f"2. multicolor_dataset/raw_images_part2.zip")

if __name__ == "__main__":
    extract_frames()
