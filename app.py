from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Paths
haarcascade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "TrainingImageLabel/Trainner.yml"
trainimage_path = "TrainingImage"
studentdetail_path = "studentdetails.csv"
attendance_path = "Attendance"

# Ensure directories exist
if not os.path.exists(trainimage_path):
    os.makedirs(trainimage_path)
if not os.path.exists("StudentDetails"):
    os.makedirs("StudentDetails")
if not os.path.exists(attendance_path):
    os.makedirs(attendance_path)
if not os.path.exists("TrainingImageLabel"):
    os.makedirs("TrainingImageLabel")

# Train the model
def train_model():
    try:
        # Load the face images and labels
        faces = []
        labels = []
        for root, dirs, files in os.walk(trainimage_path):
            for file in files:
                if file.endswith(".jpg"):
                    # Extract the enrollment number from the filename
                    enrollment_no = int(os.path.splitext(file)[0])
                    image_path = os.path.join(root, file)

                    # Load the image and convert it to grayscale
                    image = cv2.imread(image_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Detect faces in the image
                    face_cascade = cv2.CascadeClassifier(haarcascade_path)
                    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces_detected:
                        faces.append(gray[y:y+h, x:x+w])
                        labels.append(enrollment_no)

        # Train the model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))

        # Save the trained model
        recognizer.write(trainimagelabel_path)
        print("Model trained and saved successfully!")
    except Exception as e:
        print(f"Error training model: {e}")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Register a new student
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        enrollment_no = request.form['enrollment_no']
        student_name = request.form['student_name']

        if enrollment_no and student_name:
            # Capture live photo from webcam
            face_cascade = cv2.CascadeClassifier(haarcascade_path)
            cam = cv2.VideoCapture(0)
            time.sleep(2)  # Allow the camera to warm up

            ret, frame = cam.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    # Save the captured image
                    image_path = os.path.join(trainimage_path, f"{enrollment_no}.jpg")
                    cv2.imwrite(image_path, frame)
                    flash(f"Photo captured and saved for {student_name}.", "success")
                else:
                    flash("No face detected. Please try again.", "error")
                    cam.release()
                    return redirect(url_for('register'))

                cam.release()
                cv2.destroyAllWindows()

                # Save student details to CSV
                student_data = pd.DataFrame({
                    'Enrollment No': [enrollment_no],
                    'Name': [student_name]
                })
                if os.path.exists(studentdetail_path):
                    student_data.to_csv(studentdetail_path, mode='a', header=False, index=False)
                else:
                    # Create the CSV file with headers if it doesn't exist
                    student_data.to_csv(studentdetail_path, index=False)

                flash(f"Student {student_name} registered successfully!", "success")
                return redirect(url_for('register'))
            else:
                flash("Failed to capture image. Please try again.", "error")
        else:
            flash("Please fill in all fields.", "error")

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
            # Check if studentdetails.csv exists and has the correct columns
            if not os.path.exists(studentdetail_path):
                flash("Student details file not found. Please register students first.", "error")
                return redirect(url_for('attendance'))

            df = pd.read_csv(studentdetail_path)
            if 'Enrollment No' not in df.columns or 'Name' not in df.columns:
                flash("Student details file is missing required columns. Please check the file.", "error")
                return redirect(url_for('attendance'))

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                recognizer.read(trainimagelabel_path)
            except:
                flash("Model not found. Please train the model first.", "error")
                return redirect(url_for('attendance'))

            face_cascade = cv2.CascadeClassifier(haarcascade_path)
            cam = cv2.VideoCapture(0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            col_names = ["Enrollment", "Name"]
            attendance = pd.DataFrame(columns=col_names)
            
            # Keep track of marked students to prevent duplicates
            marked_students = set()

            start_time = time.time()
            while time.time() - start_time < 20:  # Run for 20 seconds
                _, im = cam.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)

                for (x, y, w, h) in faces:
                    Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    if conf < 70:
                        # Only mark attendance if student hasn't been marked yet
                        if Id not in marked_students:
                            student_name = df.loc[df["Enrollment No"] == Id]["Name"].values[0]
                            attendance.loc[len(attendance)] = [Id, student_name]
                            marked_students.add(Id)  # Add student to marked set
                        
                        # Draw green rectangle for recognized faces
                        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 260, 0), 4)
                        cv2.putText(im, str(student_name), (x+h, y), font, 1, (255, 255, 0), 4)
                    else:
                        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 25, 255), 7)
                        cv2.putText(im, "Unknown", (x+h, y), font, 1, (0, 25, 255), 4)

                cv2.imshow("Filling Attendance...", im)
                if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                    break

            cam.release()
            cv2.destroyAllWindows()

            # Save attendance to CSV only if there are entries
            if not attendance.empty:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                attendance_file = os.path.join(attendance_path, f"{subject}_{timestamp}.csv")
                attendance.to_csv(attendance_file, index=False)
                flash(f"Attendance for {subject} filled successfully! {len(marked_students)} students marked present.", "success")
            else:
                flash("No students were detected during the attendance session.", "warning")

        except Exception as e:
            flash(f"Error: {str(e)}", "error")

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