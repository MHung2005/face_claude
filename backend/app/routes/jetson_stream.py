import cv2
from flask import Blueprint, Response, stream_with_context

jetson_stream_bp = Blueprint('jetson_stream', __name__)

RTSP_URL = "rtsp://admin:Chimai@2026@10.10.10.64:554/Streaming/Channels/101"


def get_camera_frame():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

# MJPEG stream generator
def mjpeg_stream():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    finally:
        cap.release()

@jetson_stream_bp.route('/jetson/stream')
def jetson_mjpeg_stream():
    return Response(stream_with_context(mjpeg_stream()), mimetype='multipart/x-mixed-replace; boundary=frame')

@jetson_stream_bp.route('/jetson/frame', methods=['GET'])
def get_jetson_frame():
    frame = get_camera_frame()
    if frame is None:
        return Response("Không lấy được frame", status=503)
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return Response("Lỗi mã hóa JPEG", status=500)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')