import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

# 페이지 설정
st.set_page_config(page_title="AI 운동 칼로리 카운터", layout="wide")

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# UI 구성
st.title("🏋️ AI 실시간 운동 카운터 & 칼로리 계산기")
st.write("영상을 업로드하면 AI가 스쿼트 횟수를 세고 소모 칼로리를 알려줍니다.")

# 사이드바 사용자 설정
st.sidebar.header("설정")
weight = st.sidebar.number_input("몸무게(kg) 입력", value=70)
st.sidebar.info("스쿼트 1개당 약 0.4~0.5 kcal가 소모되도록 설계되었습니다.")

uploaded_file = st.file_uploader("운동 영상 업로드 (mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # 상태 변수
    counter = 0
    stage = None
    
    # 결과 표시 레이아웃
    st_frame = st.empty()
    res_col1, res_col2 = st.columns(2)
    count_text = res_col1.metric("운동 횟수", "0 회")
    cal_text = res_col2.metric("소모 칼로리", "0.00 kcal")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 변환 및 처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # 왼쪽 골반, 무릎, 발목 좌표 (각도 계산용)
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)

            # 스쿼트 카운팅 로직
            if angle > 160:
                stage = "up"
            if angle < 90 and stage == 'up':
                stage = "down"
                counter += 1
            
            # 칼로리 공식: 몸무게 반영 (개당 소모량 약산식)
            calories_burned = counter * (weight * 0.006)

            # 화면 데이터 업데이트
            count_text.metric("운동 횟수", f"{counter} 회")
            cal_text.metric("소모 칼로리", f"{calories_burned:.2f} kcal")

            # 영상 위에 랜드마크 그리기
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except Exception as e:
            pass

        # 영상 출력
        st_frame.image(image, channels="RGB", use_column_width=True)

    cap.release()
    st.balloons()
    st.success(f"운동 종료! 총 {counter}회 성공, {calories_burned:.2f} kcal 소모하셨습니다!")
