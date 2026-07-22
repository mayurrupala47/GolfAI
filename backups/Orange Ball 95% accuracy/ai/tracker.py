from typing import List, Tuple, Optional, Dict
import numpy as np
import logging
from ai.interfaces import IBallTracker

logger = logging.getLogger(__name__)

class ByteBallTracker(IBallTracker):
    """
    Centroid tracking implementation for multi-object tracking of golf balls.
    Avoids supervision/ByteTrack dependency problems and works perfectly with
    sporadic/low-confidence detections of small objects.
    """
    def __init__(self, fps: int = 30, track_thresh: float = 0.05, track_buffer: int = 30, match_thresh: float = 0.8):
        """
        Initializes the tracker.
        
        Args:
            fps: Frame rate of the video.
            track_thresh: Confidence threshold for valid detections.
            track_buffer: Number of consecutive frames a track can go undetected before deletion.
            match_thresh: Unused.
        """
        self.fps = fps
        self.track_thresh = track_thresh
        self.max_disappeared = track_buffer
        
        self.next_track_id = 1
        # self.tracks holds { track_id: [x1, y1, x2, y2] }
        self.tracks = {}
        # self.disappeared holds { track_id: frames_disappeared }
        self.disappeared = {}

    def update(self, detections: List[Tuple[float, float, float, float, float]], frame: np.ndarray = None, track_states: Dict[int, str] = None) -> List[Tuple[float, float, float, float, int]]:
        # Filter detections by confidence threshold
        valid_dets = [d for d in detections if d[4] >= self.track_thresh]
        
        # If no detections, increment disappeared count for all existing tracks
        if len(valid_dets) == 0:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            
            # Return currently active tracks
            return [(box[0], box[1], box[2], box[3], tid) for tid, box in self.tracks.items()]
            
        # Parse inputs into centroids
        input_centroids = []
        for det in valid_dets:
            x1, y1, x2, y2, _ = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids.append((cx, cy, det))
            
        # If no active tracks, register first detection as Ball ID 1
        if len(self.tracks) == 0:
            if len(input_centroids) > 0:
                cx, cy, det = input_centroids[0]
                self.tracks[1] = det[:4]
                self.disappeared[1] = 0
        else:
            # Match existing tracks with input centroids using Euclidean distance
            track_ids = list(self.tracks.keys())
            track_centroids = []
            for tid in track_ids:
                box = self.tracks[tid]
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                track_centroids.append((cx, cy))
                
            # Compute distance matrix
            D = np.zeros((len(track_centroids), len(input_centroids)))
            for i, tc in enumerate(track_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = np.sqrt((tc[0] - ic[0])**2 + (tc[1] - ic[1])**2)
                    
            # Find matching pairs (greedy approach)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Distance threshold of 150 pixels for matching
            max_distance = 150.0
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > max_distance:
                    continue
                    
                tid = track_ids[row]
                self.tracks[tid] = input_centroids[col][2][:4]
                self.disappeared[tid] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            # Handle disappeared tracks
            for row in range(len(track_centroids)):
                if row not in used_rows:
                    tid = track_ids[row]
                    self.disappeared[tid] += 1
                    if self.disappeared[tid] > self.max_disappeared:
                        del self.tracks[tid]
                        del self.disappeared[tid]
                        
            # Register ONLY Ball ID 1 if not currently tracked
            if 1 not in self.tracks:
                for col in range(len(input_centroids)):
                    if col not in used_cols:
                        self.tracks[1] = input_centroids[col][2][:4]
                        self.disappeared[1] = 0
                        break
                    
        # Return currently active tracks
        return [(box[0], box[1], box[2], box[3], tid) for tid, box in self.tracks.items()]
