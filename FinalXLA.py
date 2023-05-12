import cv2
import numpy as np
import face_recognition
import os
import datetime
import csv
import streamlit as st
import pickle

# Khởi tạo một từ điển rỗng để lưu thông tin người dùng đã điểm danh
tham_du = {}

# Step 1: Load images and encoding
path = "pic"
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Tạo file để lưu encoding
encodeFile = "encodingsfa.pickle"

if os.path.exists(encodeFile):
    # Tải encoding từ file pickle nếu đã tồn tại
    with open(encodeFile, "rb") as f:
        encodeListKnown = pickle.load(f)
else:
    # Tạo encoding nếu file chưa tồn tại
    encodeListKnown = [face_recognition.face_encodings(img)[0] for img in images]
    # Lưu encoding vào file pickle
    with open(encodeFile, "wb") as f:
        pickle.dump(encodeListKnown, f)

# Step 2: Attendance function
def markAttendance(name,frame):

    # Kiểm tra xem tên đã có trong từ điển hay chưa
    if name in tham_du:
        # Nếu tên đã có trong từ điển, chỉ cần cập nhật thông tin lần gần nhất
        tham_du[name] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Nếu tên chưa có trong từ điển, thêm thông tin mới vào
        tham_du[name] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('thamdu.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, tham_du[name]])
        cv2.imwrite(f"picSuccess/{name}.jpg", frame)

# Step 3: Streamlit app
def main():
    st.title("Ứng dụng điểm danh bằng nhận dạng khuôn mặt")
    menu = ["Trang chủ", "Điểm danh"]
    choice = st.sidebar.selectbox("Chọn chức năng", menu)

    if choice == "Trang chủ":
        st.subheader("Trang chủ")
        st.write("Chào mừng đến với ứng dụng điểm danh bằng nhận dạng khuôn mặt")

        st.image('machineLearning3.png')

        st.write('Thành viên:')
        st.write('Lê Trương Ngọc Hải - 20110465')
        st.write('Phạm Hồng Hiệu -  20110483')
        st.write('Phạm Văn Lương - 20110520')

    elif choice == "Điểm danh":
        st.subheader("Điểm danh")
        radio_choice = st.sidebar.radio("Lựa chọn điểm danh",
                                        ("Điểm danh bằng Webcam", "Điểm danh bằng thiết bị ngoại vi"))
        if radio_choice == "Điểm danh bằng Webcam":
            cap = cv2.VideoCapture(0)
        elif radio_choice == "Điểm danh bằng thiết bị ngoại vi":
            cap = cv2.VideoCapture(1)
        cap.set(3, 1920)
        cap.set(4, 1080)
        st.write("Hãy nhìn thẳng vào camera của bạn để bắt đầu quá trình điểm danh")
        st.write("Chú ý sau quá trình điểm danh hãy nhấn nút Q để thoát")

        frame = None
        while True:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang ảnh xám
            frame = cv2.equalizeHist(frame)  # Cân bằng sáng
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # Chuyển ảnh trở lại sang ảnh màu
            framS = cv2.resize(frame, (0, 0), None, fx=0.5, fy=0.5)
            framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(framS)
            encodesCurFrame = face_recognition.face_encodings(framS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if faceDis[matchIndex] < 0.4:
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name, frame)
                else:
                    name = "Unknown"
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name, frame)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Hiển thị danh sách điểm danh
        st.subheader("Danh sách điểm danh")
        if len(tham_du) > 0:
            with st.expander("Xem danh sách"):
                for name, time in tham_du.items():
                    st.write(f"{name} - {time}")
                for name in tham_du:
                    st.image(f"picSuccess/{name}.jpg", caption=name, width=200)
        else:
            st.write("Chưa có thông tin điểm danh nào")
    st.sidebar.text("Cảm ơn bạn đã sử dụng ứng dụng của chúng tôi!")
if __name__ == '__main__':
    main()