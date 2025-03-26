import cv2
import numpy as np
import sqlite3

# Initialize face detection
# Load the face detection model (ensure the correct path)
faceDetect = cv2.CascadeClassifier(r'C:\Users\Vedant\Desktop\Face Recognition Project\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insertorupdate(Id, Name, age):
    conn = sqlite3.connect(r'C:\Users\Vedant\Desktop\Face Recognition Project\database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
    Id INTEGER PRIMARY KEY,
    Name TEXT,
    Age INTEGER
    )
    ''')
    
    # Check if the record with this ID already exists
    cursor.execute('SELECT * FROM students WHERE Id = ?', (Id,))
    record = cursor.fetchone()

    if record:
        # Update existing record if the Id exists
        conn.execute("UPDATE students SET Name = ?, Age = ? WHERE Id = ?", (Name, age, Id))
    else:
        # Insert new record if the Id does not exist
        conn.execute("INSERT INTO students (Id, Name, Age) values(?, ?, ?)", (Id, Name, age))
    
    conn.commit()
    conn.close()

# Input for student details
Id = input('Enter User Id: ')
Name = input('Enter User Name: ')
age = input('Enter User Age: ')

insertorupdate(Id, Name, age)

sampleNum = 0
while True:
    ret, img = cam.read()  # Capture frame
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = faceDetect.detectMultiScale(gray,1.3,5)  # Detect faces
    
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"dataset/user.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around face
        cv2.waitKey(100)
    
    cv2.imshow("Face", img)
    
    if sampleNum > 20:  # Stop after capturing 20 samples
        break

cam.release()
cv2.destroyAllWindows()