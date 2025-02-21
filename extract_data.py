import os
import cv2
import mediapipe as mp
import csv

# Đường dẫn đến thư mục chứa video
data_path = r"E:\HUIT file\HK6-HUIT\Video_Training"

# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Danh sách các điểm cần lấy (bỏ qua 0 → 10) các điểm từ 0 -> 10 thuộc về khuôn mặt nên k cần thiết cho dữ liệuliệu
selected_landmarks = [i for i in range(11, 33)]  # Pose có 33 điểm

# Định nghĩa tiêu đề cho file CSV
header = []
for i in selected_landmarks:
    header.append(f"landmark_{i}_x")
    header.append(f"landmark_{i}_y")
    header.append(f"landmark_{i}_z")
header.append("label")

# Tạo file CSV (nếu chưa có)
csv_file = "data_training.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    
    # Ghi tiêu đề nếu file mới được tạo
    if not file_exists:
        writer.writerow(header)

    # Duyệt qua tất cả thư mục bài tập
    exercise_labels = sorted(os.listdir(data_path))

    for label in exercise_labels:
        folder_path = os.path.join(data_path, label)

        if not os.path.isdir(folder_path):
            continue  # Bỏ qua nếu không phải thư mục

        print(f"📂 Đọc video từ bài tập: {label}")

        for video_file in sorted(os.listdir(folder_path)):
            video_path = os.path.join(folder_path, video_file)

            # Kiểm tra xem có phải file video hợp lệ không
            if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue  

            print(f"🎥 Đang đọc video: {video_file}")

            # Mở video bằng OpenCV
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Hết video

                # Chuyển đổi sang RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    row = []
                    for i in selected_landmarks:
                        landmark = results.pose_landmarks.landmark[i]
                        row.extend([landmark.x, landmark.y, landmark.z])  # Lấy x, y, z

                    row.append(label)  # Gán nhãn là tên bài tập
                    writer.writerow(row)  # Ghi vào CSV

                # Hiển thị video
                # cv2.imshow(f"Video - {label}", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break  

            cap.release()#đóng video sau khi xử lý xong
            # cv2.destroyAllWindows()  # Đóng cửa sổ video khi chuyển sang video khác

cv2.destroyAllWindows()
