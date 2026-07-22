import os
import json
import queue
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

# Shared variables and queues
frame_queue = queue.Queue(maxsize=1)
latest_stroke_frame = None  # Stores the bytes of the latest stroke event frame
event_queues = []

@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    while True:
        try:
            # Block until a frame is pushed from the main pipeline
            frame_bytes = frame_queue.get(block=True)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception:
            break

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/stroke_frame")
def stroke_frame():
    """Returns the latest stroke event frame as JPEG."""
    global latest_stroke_frame
    if latest_stroke_frame is not None:
        return Response(latest_stroke_frame, mimetype='image/jpeg')
    return Response(status=204)  # No content yet

@app.route("/events")
def events():
    def event_stream():
        q = queue.Queue()
        event_queues.append(q)
        try:
            while True:
                # Block until an event is published
                data = q.get(block=True)
                yield f"data: {json.dumps(data)}\n\n"
        except GeneratorExit:
            event_queues.remove(q)
            
    return Response(event_stream(), mimetype="text/event-stream")

# Helper function to broadcast new frames to the web server
def push_frame(jpeg_bytes):
    # If queue is full, drop the old frame to maintain real-time low latency
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
    frame_queue.put(jpeg_bytes)

# Helper function to push stroke event frame
def push_stroke_frame(jpeg_bytes):
    global latest_stroke_frame
    latest_stroke_frame = jpeg_bytes

# Helper function to broadcast stroke event notifications to all connected browsers
def push_stroke_event(ball_id, stroke_count, event_type="stroke"):
    import time
    event_data = {
        "event": event_type,
        "ball_id": ball_id,
        "stroke_count": stroke_count,
        "timestamp": time.strftime("%H:%M:%S")
    }
    for q in list(event_queues):
        q.put(event_data)

if __name__ == "__main__":
    # Runs the web server locally on port 5001 (allowing you to connect from any device)
    app.run(host="0.0.0.0", port=5001, threaded=True)
