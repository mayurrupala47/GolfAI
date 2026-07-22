import cv2
import json
import os
import argparse
import numpy as np

class CalibrationUI:
    def __init__(self, video_path, output_json="config/calibration.json"):
        self.video_path = video_path
        self.output_json = output_json
        
        # Load video first frame
        self.cap = cv2.VideoCapture(video_path)
        ret, self.frame = self.cap.read()
        self.cap.release()
        
        if not ret:
            raise RuntimeError(f"Could not load first frame of video: {video_path}")
            
        self.orig_h, self.orig_w = self.frame.shape[:2]
        import yaml
        if os.path.exists("config/config.yaml"):
            with open("config/config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            self.proc_w = cfg.get("processing", {}).get("resize_width", 1280)
        else:
            self.proc_w = 1280
            
        self.proc_h = int(self.orig_h * (self.proc_w / self.orig_w))
        
        # Display the exact processing resolution used by the tracker to ensure 1:1 pixel parity
        self.disp_w = self.proc_w
        self.disp_h = self.proc_h
            
        # Resize reference frame once
        self.resized_frame = cv2.resize(self.frame, (self.disp_w, self.disp_h))
        
        # No scaling needed since display is exactly the processing resolution
        self.to_orig_x = 1.0
        self.to_orig_y = 1.0
        
        self.win_name = "Mini Golf AI Calibration Utility"
        cv2.namedWindow(self.win_name, cv2.WINDOW_AUTOSIZE)
        
        # State relative to display dimensions
        self.playable_area_disp = []
        self.ignore_circles_disp = []  # Tees & Cups
        self.ignore_polygons_disp = [] # Linear Pipes
        
        self.current_step = "PLAYABLE_AREA" # PLAYABLE_AREA, TEE, CUPS, PIPES
        
        self.temp_points = []
        self.temp_radius = 20 # Display radius
        self.temp_center = None
        self.mouse_cursor = (0, 0)
        
        self.instructions = [
            "STEP 1: Define Playable Area Corners",
            " - Click on the 4 corners of the playing field.",
            " - Lines will draw automatically as you click.",
            " - After the 4th click, the area will close and advance.",
            " - Press 'r' at any time to reset current step."
        ]

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_step == "PLAYABLE_AREA":
                self.temp_points.append([x, y])
                # Auto-close on 4 corners
                if len(self.temp_points) == 4:
                    self.playable_area_disp = list(self.temp_points)
                    self.temp_points = []
                    self.current_step = "TEE"
                    self.temp_center = None
                    self.instructions = [
                        "STEP 2: Define Tee Point",
                        " - Click on the center of the player tee.",
                        " - Use '+' and '-' to adjust the radius.",
                        " - Press 'Enter' when done to save and proceed."
                    ]
            elif self.current_step == "TEE":
                self.temp_center = (x, y)
            elif self.current_step == "CUPS":
                self.temp_center = (x, y)
            elif self.current_step == "PIPES":
                # Polygon masking for linear pipes
                self.temp_points.append([x, y])
                if len(self.temp_points) == 4:
                    self.ignore_polygons_disp.append({
                        "name": f"Pipes {len(self.ignore_polygons_disp) + 1}",
                        "points": list(self.temp_points)
                    })
                    self.temp_points = []

    def draw_hud(self, img):
        # Instructions overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        y_offset = 30
        for line in self.instructions:
            cv2.putText(img, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 22
            
        cv2.putText(img, f"Current Step: {self.current_step}", (20, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Draw Playable Area Polygon
        pts = np.array(self.playable_area_disp, dtype=np.int32)
        if len(pts) > 0:
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(img, tuple(p), 6, (0, 255, 0), -1)
                
        # Draw temp points of polygon/boxes
        if self.current_step in ["PLAYABLE_AREA", "PIPES"] and len(self.temp_points) > 0:
            for p in self.temp_points:
                cv2.circle(img, tuple(p), 5, (0, 165, 255), -1)
            
            # Draw lines between placed points
            pts_array = np.array(self.temp_points, dtype=np.int32)
            cv2.polylines(img, [pts_array], False, (0, 165, 255), 2)
            
            # Draw guideline line to mouse cursor
            cv2.line(img, tuple(self.temp_points[-1]), self.mouse_cursor, (0, 255, 255), 1, cv2.LINE_AA)
                
        # Draw Ignore Circles (Tees & Cups)
        for region in self.ignore_circles_disp:
            color = (0, 0, 255)
            if "tee" in region["name"].lower():
                color = (255, 255, 0)
            elif "cup" in region["name"].lower() or "hole" in region["name"].lower():
                color = (255, 0, 255)
            cv2.circle(img, (region["x"], region["y"]), region["radius"], color, 2)
            cv2.circle(img, (region["x"], region["y"]), 3, color, -1)
            cv2.putText(img, region["name"], (region["x"] - 20, region["y"] - region["radius"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Draw Ignore Polygons (Pipes)
        for poly in self.ignore_polygons_disp:
            color = (0, 0, 255) # Red for obstacles
            poly_pts = np.array(poly["points"], dtype=np.int32)
            cv2.polylines(img, [poly_pts], True, color, 2)
            for p in poly_pts:
                cv2.circle(img, tuple(p), 5, color, -1)
            cv2.putText(img, poly["name"], tuple(poly["points"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Draw currently adjusting circle (Step 2 & 3)
        if self.current_step in ["TEE", "CUPS"] and self.temp_center:
            color = (0, 165, 255)
            cv2.circle(img, self.temp_center, self.temp_radius, color, 2)
            cv2.circle(img, self.temp_center, 3, color, -1)
            cv2.putText(img, f"Radius: {self.temp_radius}px (keys: +/-)", (self.temp_center[0] - 40, self.temp_center[1] + self.temp_radius + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def run(self):
        cv2.setMouseCallback(self.win_name, self.mouse_callback)
        
        while True:
            display_img = self.resized_frame.copy()
            self.draw_hud(display_img)
            cv2.imshow(self.win_name, display_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            # Close/Exit
            if key == 27:  # ESC
                break
                
            # Reset current step temp inputs
            if key == ord('r'):
                self.temp_points = []
                self.temp_center = None
                self.temp_radius = 20
                if self.current_step == "PLAYABLE_AREA":
                    self.playable_area_disp = []
                elif self.current_step == "PIPES":
                    # Clear last added pipe polygon
                    if len(self.ignore_polygons_disp) > 0:
                        self.ignore_polygons_disp.pop()
                
            # Increase / Decrease circle radius
            if key == ord('+') or key == ord('='):
                self.temp_radius = min(self.temp_radius + 2, 200)
            if key == ord('-') or key == ord('_'):
                self.temp_radius = max(self.temp_radius - 2, 5)
                
            # Step specific keyboard commands
            if self.current_step == "TEE":
                if (key == 13 or key == 10) and self.temp_center: # Enter key
                    self.ignore_circles_disp.append({
                        "name": f"Tee {len([r for r in self.ignore_circles_disp if 'tee' in r['name'].lower()]) + 1}",
                        "x": self.temp_center[0],
                        "y": self.temp_center[1],
                        "radius": self.temp_radius
                    })
                    self.temp_center = None
                    self.temp_radius = 20
                    self.current_step = "CUPS"
                    self.instructions = [
                        "STEP 3: Define Hole/Cup Area",
                        " - Click on the center of the target cup/hole.",
                        " - Use '+' and '-' to adjust the radius.",
                        " - Press 'a' to add another cup/hole.",
                        " - Press 'Enter' when done with all cups."
                    ]
                    
            elif self.current_step == "CUPS":
                if key == ord('a') and self.temp_center:
                    self.ignore_circles_disp.append({
                        "name": f"Cup {len([r for r in self.ignore_circles_disp if 'cup' in r['name'].lower() or 'hole' in r['name'].lower()]) + 1}",
                        "x": self.temp_center[0],
                        "y": self.temp_center[1],
                        "radius": self.temp_radius
                    })
                    self.temp_center = None
                    self.temp_radius = 20
                elif (key == 13 or key == 10): # Enter
                    if self.temp_center:
                        self.ignore_circles_disp.append({
                            "name": f"Cup {len([r for r in self.ignore_circles_disp if 'cup' in r['name'].lower() or 'hole' in r['name'].lower()]) + 1}",
                            "x": self.temp_center[0],
                            "y": self.temp_center[1],
                            "radius": self.temp_radius
                        })
                    self.temp_center = None
                    self.temp_radius = 20
                    self.temp_points = []
                    self.current_step = "PIPES"
                    self.instructions = [
                        "STEP 4: Define Pipes/Obstacle Polygon",
                        " - Click 4 points to draw a box outlining the pipes.",
                        " - Lines connect automatically. Auto-closes on 4th click.",
                        " - Press 'a' to confirm and start another obstacle box.",
                        " - Press 'Enter' when fully finished to save."
                    ]
                    
            elif self.current_step == "PIPES":
                if (key == 13 or key == 10): # Enter key
                    break

        cv2.destroyAllWindows()
        self.save_calibration()

    def save_calibration(self):
        if len(self.playable_area_disp) == 0:
            print("No playable area defined. Aborting save.")
            return
            
        # Scale playable area back to original resolution
        playable_area_orig = [
            [int(p[0] * self.to_orig_x), int(p[1] * self.to_orig_y)]
            for p in self.playable_area_disp
        ]
        
        # Scale ignore regions back to original resolution
        ignore_regions_orig = []
        
        # Add scaled circles (Tees & Cups)
        for region in self.ignore_circles_disp:
            ignore_regions_orig.append({
                "name": region["name"],
                "type": "circle",
                "x": int(region["x"] * self.to_orig_x),
                "y": int(region["y"] * self.to_orig_y),
                "radius": int(region["radius"] * self.to_orig_x)
            })
            
        # Add scaled polygons (Pipes / Obstacles)
        for poly in self.ignore_polygons_disp:
            pts_orig = [
                [int(p[0] * self.to_orig_x), int(p[1] * self.to_orig_y)]
                for p in poly["points"]
            ]
            ignore_regions_orig.append({
                "name": poly["name"],
                "type": "polygon",
                "points": pts_orig
            })
            
        data = {
            "source_resolution": [self.proc_w, self.proc_h],
            "playable_area": playable_area_orig,
            "ignore_regions": ignore_regions_orig
        }
        
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved new calibration layout to {self.output_json}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Mini Golf AI Calibration UI Tool")
    parser.add_argument("--video", type=str, required=True, help="Path to video file for reference frame.")
    parser.add_argument("--output", type=str, default="config/calibration.json", help="Path to output calibration JSON.")
    args = parser.parse_args()
    
    ui = CalibrationUI(args.video, args.output)
    ui.run()
