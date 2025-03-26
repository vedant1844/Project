import cv2
import numpy as np
import sqlite3

# Verify path to the cascade file
cascade_path = r'C:\Users\Vedant\Desktop\Face Recognition Project\haarcascade_frontalface_default.xml'
facedetect = cv2.CascadeClassifier(cascade_path)

# Ensure the cascade file is loaded correctly
if facedetect.empty():
    print("Error: Unable to load the cascade classifier.")
    exit()

cam = cv2.VideoCapture(0)

# Create the face recognizer and load the trained data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer/trainingdata.yml")

# Function to initialize the database and create the table if not already created
def initialize_database():
    conn = sqlite3.connect("sqlite.db")
    conn.execute('''
    CREATE TABLE IF NOT EXISTS STUDENTS (
        Id INTEGER PRIMARY KEY,
        Name TEXT,
        Age INTEGER
    )
    ''')
    conn.commit()
    conn.close()

# Function to get student profile by ID
def getprofile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE Id =?", (id,))
    profile = cursor.fetchone()
    print(f"Profile fetched for ID {id}: {profile}")  # Debugging statement
    conn.close()
    return profile

# Initialize the database only once
initialize_database()

while(True):
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        print(f"Predicted ID: {id}, Confidence: {conf}")  # Debugging statement
        
        profile = getprofile(id)
        
        if profile is not None:
            cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
            cv2.putText(img, "Age: " + str(profile[2]), (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
        else:
            cv2.putText(img, "Name: Unknown", (x, y + h + 28), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "Age: Unknown", (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("FACE", img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
