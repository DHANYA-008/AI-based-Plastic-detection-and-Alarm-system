
from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import winsound
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("model/best.pt")

# Open webcam
camera = cv2.VideoCapture(0)

alarm_cooldown = 0  # to avoid continuous alarm

def generate_frames():
    global alarm_cooldown

    while True:
        success, frame = camera.read()
        if not success:
            break

        # YOLO detection
        results = model(frame, conf=0.5)

        plastic_detected = False

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label.lower() == "plastic":
                    plastic_detected = True

        # Alarm logic (once every 5 seconds)
        if plastic_detected and time.time() - alarm_cooldown > 5:
            winsound.PlaySound("alarm.wav", winsound.SND_FILENAME)
            alarm_cooldown = time.time()

        # Draw boxes
        annotated_frame = results[0].plot()

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
