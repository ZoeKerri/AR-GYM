import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model("exercise_cnn_model.h5")  # Thay "model.h5" bằng tên file mô hình của bạn

# Danh sách bài tập tương ứng với nhãn mô hình
exercise_labels = ["barbell biceps curl", "hammer curl", "push-up", "shoulder press", "squat"]

# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Danh sách các điểm cần lấy (bỏ qua các điểm khuôn mặt 0 -> 10)
selected_landmarks = [i for i in range(11, 33)]  # Mediapipe Pose có 33 điểm

# Mở camera
cap = cv2.VideoCapture(0)  # 0 là webcam mặc định

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi sang RGB (Mediapipe yêu cầu ảnh RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dự đoán pose
    results = pose.process(image)

    predicted_text = "None"  # Mặc định là "None" nếu không có dự đoán hợp lệ

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

        if 0 <= predicted_label < len(exercise_labels):
            predicted_text = exercise_labels[predicted_label]

    # Hiển thị kết quả lên màn hình
    cv2.putText(frame, f"Predicted: {predicted_text}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Hiển thị video
    cv2.imshow("Camera Prediction", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
