import tkinter as tk
from tkinter import Message, Text
import tkinter.ttk as ttk
import tkinter.font as font

import cv2
import os
import shutil
import csv
import numpy as np
import pandas as pd

from PIL import Image, ImageTk

import datetime
import time

from urllib.request import urlopen
from ssl import SSLContext, PROTOCOL_TLSv1

website = tk.Tk()
website.title(" *** Employee Attendance Management System via Face Recognition *** ")
dialog_title = 'QUIT'
website.geometry('2200x1500')
website.configure()  

#website.attributes('-fullscreen', True)

website.grid_rowconfigure(0, weight=1)
website.grid_columnconfigure(0, weight=1)

path = "facer2.jpg" # background image

# Creates a Tkinter-compatible photo image
img = ImageTk.PhotoImage(Image.open(path))


# The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = tk.Label(website, image=img)
panel.pack(side="left", fill="x", expand="no")

message = tk.Label(website, text="Employee-Face-Recognition-Attendance-System", bg="darkblue",
                   fg="black", width=55, height=2, font=('Algerian', 30, 'italic bold underline'))
message.place(x=80, y=20)

#Employee ID label
label1 = tk.Label(website, text="Employee ID", width=20, height=2,
               fg="black", bg="#00ffff", font=('Cambria', 16, ' bold '))
label1.place(x=400, y=200)

#Enter corresponding Employee ID
text1 = tk.Entry(website, width=20, bg="#008080", 
               fg="white", font=('Cambria', 16, ' bold '))
text1.place(x=700, y=215)

#Employee Name label
label2 = tk.Label(website, text="Employee Name", width=20, fg="black",
                bg="#00ffff", height=2, font=('Cambria', 16, ' bold '))
label2.place(x=400, y=300)

#Enter corresponding employee name
text2 = tk.Entry(website, width=20, bg="#008080",
                fg="white", font=('Cambria', 16, ' bold '))
text2.place(x=700, y=315)

#Notification label
label3 = tk.Label(website, text="Notification :", width=20, fg="black",
                bg="#00ffff", height=2, font=('Cambria', 16, ' bold '))
label3.place(x=400, y=400)

#Notification of 'image trained or not' display box
message1 = tk.Label(website, text="", bg="#008080", fg="white", width=50,
                   height=2, activebackground="yellow", font=('Cambria', 16, ' bold '))
message1.place(x=700, y=400)

#Attendance label
label4 = tk.Label(website, text="Attendance : ", width=20, fg="black",
                bg="#00ffff", height=2, font=('Cambria', 16, ' bold '))
label4.place(x=400, y=650)

# database result display box
message2 = tk.Label(website, text="", fg="white", bg="#008080",
                    activeforeground="green", width=50, height=4, font=('Cambria', 16, ' bold '))
message2.place(x=700, y=650)

def clear1():
    text1.delete(0, 'end')
    result1 = ""
    message1.configure(text=result1)


def clear2():
    text2.delete(0, 'end')
    result2 = ""
    message1.configure(text=result2)


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(x)
        return True
    except (TypeError, ValueError):
        pass

    return False

#function to capture images via webcam
def TakeImages():
    Id = (text1.get())
    name = (text2.get())
    if(is_number(Id) and name.isalpha()):
        camera = cv2.VideoCapture(0)

        #face recognition algorithm used is haarcascade having an accuracy of approxiamately 97%
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)

        #Variable to store snaps of image captured
        sampleNo = 0
        while(True):
            r, img = camera.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 4)
            
            # Convert frame to grayscale
            #gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

            # Detect frames of different sizes, list of faces rectangles
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNo = sampleNo+1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name + "."+Id + '.' +
                            str(sampleNo) + ".jpg", gray[y:y+h, x:x+w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNo > 100:
                break
        camera.release()
        cv2.destroyAllWindows()
        result = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]

        #appending records to csv file
        with open('EmployeeDetails\EmployeeDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=result)

    else:
        if(is_number(Id)):
            result = "Enter Alphabetical Name"
            message.configure(text=result)

        if(name.isalpha()):
            result = "Enter Numeric Id"
            message.configure(text=result)

#Recognizer used for images is LBPHFaceRecognizer
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    
    harcascadePath = "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(harcascadePath)

    faces, Id = getImagesAndLabels("TrainingImage")

    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    result = "Image Trained"  
    message1.configure(text=result)


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    faces = []
    
    Ids = []
    # now looping through all the image paths and loading the Ids and the corresponding images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageArray = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageArray)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    
    #Recognizer used for recognising snaps  : LBPHFaceRecognizer
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"

    #haarcascade classifier used for classification : 97% accuracy approxiamately
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    dataset = pd.read_csv("EmployeeDetails\EmployeeDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    column_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=column_names)

    while True:
        r , im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x, y, w, h) in face:
            cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
            Id, config = recognizer.predict(gray[y:y+h, x:x+w])

            if(config < 50):
                t = time.time()
                date = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    t).strftime('%H:%M:%S')
                data = dataset.loc[dataset['Id'] == Id]['Name'].values
                tt = str(Id)+"-"+data
                attendance.loc[len(attendance)] = [Id,data, date, timeStamp]

            else:
                Id = 'Unknown'
                tt = str(Id)
            if(config > 75):
                File = len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(File) +
                            ".jpg", im[y:y+h, x:x+w])
            cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        
        if (cv2.waitKey(1) == ord('q')):
            break

    t = time.time()
    date = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(t).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()

    # cv2.waitKey
    cv2.destroyAllWindows()
    # print(attendance)
    res = attendance
    message2.configure(text=res)


#clear the contents in Employee ID field
clearButton1 = tk.Button(website, text="Clear", command=clear1, fg="#e6b800", bg="#00334d",
        width=20, height=2, activebackground="Red", font=('Cambria', 16, ' bold '))
clearButton1.place(x=1000, y=200)

#clear the contents in Employee Name field
clearButton2 = tk.Button(website, text="Clear", command=clear2, fg="#e6b800", bg="#00334d",
             width=20, height=2, activebackground="Red", font=('Cambria', 16, ' bold '))
clearButton2.place(x=1000, y=300)

#Capture snaps of images
takeImage = tk.Button(website, text="Capture Images", command=TakeImages, fg="red", bg="#330026",
                    width=20, height=3, activebackground="Red", font=('Cambria', 16, ' bold '))
takeImage.place(x=200, y=500)

#Click to Train captured images
trainImages = tk.Button(website, text="Train Images", command=TrainImages, fg="red",
                     bg="#330026", width=20, height=3, activebackground="Red", font=('Cambria', 16, ' bold '))
trainImages.place(x=500, y=500)

#Button to track trained images
trackImage= tk.Button(website, text="Track Images", command=TrackImages, fg="red",
                     bg="#330026", width=20, height=3, activebackground="Red", font=('Cambria', 16, ' bold '))
trackImage.place(x=800, y=500)




#Exit from the window
quitButton = tk.Button(website, text="Quit", command=quit, fg="red", bg="#330026",
                       width=20, height=3, activebackground="Red", font=('Cambria', 16, ' bold '))
quitButton.place(x=1100, y=500)


website.mainloop()
