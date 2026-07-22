import cv2
import os
import sys
import time

def main():
    source = "tcp://10.30.7.125:5000"
    output_filename = "recorded_golf_course.mp4"
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]
        
    print(f"Connecting to stream: {source}...")
    
    # Draining OpenCV's FFMPEG buffer settings
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot connect to stream {source}")
        print("Please make sure rpicam-vid is actively running on the Raspberry Pi.")
        return
        
    # Get frame properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 100:
        fps = 30.0
        
    print(f"Connected successfully! Stream resolution: {w}x{h} @ {fps:.1f} FPS")
    print(f"Recording to file: {output_filename}")
    print("Press 'q' or Ctrl+C in this terminal to STOP recording.")
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\nStream ended or lost.")
                break
                
            # Write frame to video file
            out.write(frame)
            frame_count += 1
            
            # Show live preview so they can see what is being captured
            cv2.imshow("Recording Stream - Press 'q' to Stop", frame)
            
            # Calculate dynamic statistics
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\rCaptured: {frame_count} frames | Elapsed: {elapsed:.1f}s | FPS: {current_fps:.1f}")
            sys.stdout.flush()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\nSUCCESS: Video saved!")
        print(f"Total frames written: {frame_count}")
        print(f"File path: {os.path.abspath(output_filename)}")
        print(f"Duration: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
