import cv2
from fer import FER
import numpy as np

# 웹캠 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# FER 인스턴스 생성
detector = FER(mtcnn=True)

# 여러 프레임에서 감정 결과를 저장할 리스트
emotion_history = []

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 프레임 전처리 (그레이스케일 변환)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.equalizeHist(gray_frame)
    color_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # 표정 인식
    result = detector.detect_emotions(color_frame)

    # 검출된 얼굴과 표정 그리기
    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        max_emotion = max(emotions, key=emotions.get)

        # 얼굴 주위에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 표정 텍스트 추가
        text = f"{max_emotion}: {emotions[max_emotion]:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 감정 결과를 히스토리에 추가
        emotion_history.append(max_emotion)

    # 감정 히스토리에서 가장 빈번한 감정을 표시
    if len(emotion_history) > 10:  # 마지막 10 프레임을 고려
        emotion_history = emotion_history[-10:]
        most_common_emotion = max(set(emotion_history), key=emotion_history.count)
        cv2.putText(frame, f"Most common emotion: {most_common_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면에 표시
    cv2.imshow('Webcam - Facial Expression Recognition', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠과 윈도우 정리
cap.release()
cv2.destroyAllWindows()
