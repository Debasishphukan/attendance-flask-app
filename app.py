from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")

# Paths
haarcascade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel/Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "studentdetails.csv"
attendance_path = "Attendance"

# Ensure directories exist
for path in [trainimage_path, "StudentDetails", attendance_path, "TrainingImageLabel"]:
    if not os.path.exists(path):
        os.makedirs(path)

# Train the model
def train_model():
    try:
        faces = []
        labels = []
        for root, dirs, files in os.walk(trainimage_path):
            for file in files:
                if file.endswith(".jpg"):
                    enrollment_no = int(os.path.splitext(file)[0])
                    image_path = os.path.join(root, file)

                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    face_cascade = cv2.CascadeClassifier(haarcascade_path)
                    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces_detected:
                        faces.append(gray[y:y+h, x:x+w])
                        labels.append(enrollment_no)

        if faces and labels:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(labels))
            recognizer.write(trainimagelabel_path)
            print("Model trained and saved successfully!")
        else:
            print("No faces found to train the model.")
    except Exception as e:
        print(f"Error training model: {e}")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Capture page (new)
@app.route('/capture')
def capture():
    return render_template('capture.html')

# Upload photo from capture.html (new)
@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    try:
        data = request.files['photo']
        enrollment_no = request.form['enrollment_no']
        student_name = request.form['student_name']

        if enrollment_no and student_name and data:
            img_path = os.path.join(trainimage_path, f"{enrollment_no}.jpg")
            data.save(img_path)

            student_data = pd.DataFrame({
                'Enrollment No': [enrollment_no],
                'Name': [student_name]
            })
            if os.path.exists(studentdetail_path):
                student_data.to_csv(studentdetail_path, mode='a', header=False, index=False)
            else:
                student_data.to_csv(studentdetail_path, index=False)

            flash(f"Photo uploaded and student {student_name} registered successfully!", "success")
        else:
            flash("Missing enrollment number, name, or photo.", "error")

    except Exception as e:
        flash(f"Error uploading photo: {str(e)}", "error")

    return redirect(url_for('home'))

# Register page (updated: no webcam capture here anymore)
@app.route('/register', methods=['GET'])
def register():
    return render_template('register.html')

# Train the model
@app.route('/train', methods=['GET'])
def train():
    train_model()
    flash("Model trained successfully!", "success")
    return redirect(url_for('register'))

# Take attendance
@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        subject = request.form.get('subject')
        if not subject:
            flash("Please enter the subject name.", "error")
            return redirect(url_for('attendance'))

        try:
            if not os.path.exists(studentdetail_path):
                flash("Student details file not found. Please register students first.", "error")
                return redirect(url_for('attendance'))

            df = pd.read_csv(studentdetail_path)
            if 'Enrollment No' not in df.columns or 'Name' not in df.columns:
                flash("Student details file is missing required columns.", "error")
                return redirect(url_for('attendance'))

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(trainimagelabel_path)

            face_cascade = cv2.CascadeClassifier(haarcascade_path)
            cam = cv2.VideoCapture(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            attendance_df = pd.DataFrame(columns=["Enrollment", "Name"])
            marked_students = set()

            start_time = time.time()
            while time.time() - start_time < 20:
                _, frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    if conf < 70 and Id not in marked_students:
                        student_name = df.loc[df["Enrollment No"] == Id]["Name"].values[0]
                        attendance_df.loc[len(attendance_df)] = [Id, student_name]
                        marked_students.add(Id)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                        cv2.putText(frame, str(student_name), (x, y-10), font, 0.8, (255, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), font, 0.8, (0, 0, 255), 2)

                cv2.imshow("Attendance", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            cam.release()
            cv2.destroyAllWindows()

            if not attendance_df.empty:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                attendance_file = os.path.join(attendance_path, f"{subject}_{timestamp}.csv")
                attendance_df.to_csv(attendance_file, index=False)
                flash(f"Attendance for {subject} completed! {len(marked_students)} students marked present.", "success")
            else:
                flash("No students were detected during the attendance session.", "warning")

        except Exception as e:
            flash(f"Error during attendance: {str(e)}", "error")

    return render_template('attendance.html')

# View attendance
@app.route('/view_attendance')
def view_attendance():
    attendance_files = [f for f in os.listdir(attendance_path) if f.endswith('.csv')]
    attendance_data = []
    for file in attendance_files:
        df = pd.read_csv(os.path.join(attendance_path, file))
        attendance_data.append({
            'subject': file.split('_')[0],
            'date': file.split('_')[1],
            'data': df.to_dict('records')
        })
    return render_template('view_attendance.html', attendance_data=attendance_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
