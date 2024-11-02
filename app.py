from flask import Flask, request, render_template_string, url_for
from ultralytics import YOLO
import tempfile
import cv2
from moviepy.editor import VideoFileClip
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static')  # 절대 경로로 설정

# HTML 템플릿 (원본 및 검출된 비디오 재생을 위한 비디오 태그 추가)
HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>비디오 객체 검출 및 재인코딩</title>
</head>
<body>
  <h1>YOLO 모델과 비디오 파일 업로드</h1>
  <form action="/process" method="post" enctype="multipart/form-data">
    <label for="model">YOLO 모델 업로드 (.pt 파일):</label>
    <input type="file" name="model" required>
    <br><br>
    <label for="video">비디오 파일 업로드 (.mp4, .mov, .avi 파일):</label>
    <input type="file" name="video" required>
    <br><br>
    <input type="submit" value="검출 실행">
  </form>

  {% if original_video_url %}
  <h2>업로드한 원본 비디오</h2>
  <video width="640" height="480" controls autoplay>
    <source src="{{ original_video_url }}" type="video/mp4">
    사용 중인 브라우저는 비디오 태그를 지원하지 않습니다.
  </video>
  {% endif %}

  {% if processed_video_url %}
  <h2>사물 검출 후 결과 비디오</h2>
  <video width="640" height="480" controls>
    <source src="{{ processed_video_url }}" type="video/mp4">
    사용 중인 브라우저는 비디오 태그를 지원하지 않습니다.
  </video>
  {% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/process", methods=["POST"])
def process_video():
    # 모델 파일과 비디오 파일 받기
    model_file = request.files.get("model")
    video_file = request.files.get("video")

    if not model_file or not video_file:
        return "모델 파일과 비디오 파일을 모두 업로드해야 합니다.", 400

    # 모델 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model:
        temp_model.write(model_file.read())
        model_path = temp_model.name

    # YOLO 모델 로드
    model = YOLO(model_path)

    # static 폴더가 없으면 생성
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # 원본 비디오 파일 저장 경로 설정
    original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(original_video_path)

    # 원본 비디오의 URL 생성
    original_video_url = url_for('static', filename=os.path.basename(original_video_path))

    # 검출 결과 저장할 임시 비디오 파일 경로
    output_path = original_video_path.replace(".mp4", "_output.mp4")

    # 비디오 처리 시작
    cap = cv2.VideoCapture(original_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 프레임별로 객체 검출 수행
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 예측 수행
        results = model(frame)
        detections = results[0].boxes if len(results) > 0 else []

        # 검출된 객체에 대해 바운딩 박스 그리기
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # moviepy를 사용하여 재인코딩
    reencoded_path = output_path.replace(".mp4", "_reencoded.mp4")
    clip = VideoFileClip(output_path)
    clip.write_videofile(reencoded_path, codec="libx264", audio_codec="aac")

    # 검출된 비디오의 URL 생성
    processed_video_url = url_for('static', filename=os.path.basename(reencoded_path))

    # 원본 및 검출된 비디오 URL을 HTML 템플릿에 전달하여 렌더링
    return render_template_string(HTML_TEMPLATE, original_video_url=original_video_url, processed_video_url=processed_video_url)

if __name__ == "__main__":
    app.run(debug=True)
