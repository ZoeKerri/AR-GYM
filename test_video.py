import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model("exercise_cnn_model.h5")  # Đổi thành tên file mô hình của bạn

# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Danh sách các điểm cần lấy (bỏ qua các điểm khuôn mặt 0 -> 10)
selected_landmarks = [i for i in range(11, 33)]  # Mediapipe Pose có 33 điểm

# Đọc video từ file
video_path = "shoulder press_1.mp4"  # Đổi thành tên file video của bạn
cap = cv2.VideoCapture(video_path)

# Lưu video đầu ra (nếu cần)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec để lưu video
out = cv2.VideoWriter("output_video2.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(3)), int(cap.get(4))))  # Lưu video với cùng độ phân giải

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hết video thì thoát

    # Chuyển đổi sang RGB (Mediapipe yêu cầu ảnh RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dự đoán pose
    results = pose.process(image)

    if results.pose_landmarks:
        row = []
        for i in selected_landmarks:
            landmark = results.pose_landmarks.landmark[i]
            row.extend([landmark.x, landmark.y, landmark.z])  # Lấy tọa độ x, y, z của từng điểm

        # Chuyển thành mảng NumPy và reshape để đưa vào mô hình
        input_data = np.array(row).reshape(1, -1)

        # Dự đoán nhãn
        prediction = model.predict(input_data)
        predicted_label = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất

        # Hiển thị kết quả lên màn hình
        cv2.putText(frame, f"Predicted: {predicted_label}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị video
    cv2.imshow("Video Prediction", frame)

    # Ghi vào video đầu ra (nếu cần)
    out.write(frame)

    # Nhấn 'q' để thoát sớm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
