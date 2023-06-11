import tkinter as tk
from PIL import ImageTk, Image
import cv2
import time
from keras.models import model_from_json
import numpy as np
import pandas as pd
from cvzone.FaceMeshModule import FaceMeshDetector
import multiprocessing as mp

fieldnames = ["x_Time", "y_EmoSum"]
x_Time = []
y_Emotion = []
y_EmoSum = 0

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

ratioList = []
blinkList = []

idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

blinkCounter = 0
counter = 0
final_count = 0
timed_blink = 0
avg_blinks = 0
color = (255, 0, 255)

lie_Counter = 0
truth_Counter = 0
neutral_Counter = 0

# **************************GUI****************************

# root = tk.Tk()
# root.minsize(1440, 750)
# root.attributes("-fullscreen", True)
# root.attributes("-topmost", True)
# root.attributes("-toolwindow", True)

# Create a frame
# app = tk.Frame(root)
# app.grid()
# app.rowconfigure(0, weight=9)
# app.rowconfigure(1, weight=1)
# app.columnconfigure(0, weight=1)
# app.columnconfigure(1, weight=1)

# videoFrame = tk.Frame(app, bg='black')
# videoFrame.grid(row=1, column=0)

# graphFrame = tk.Frame(app, bg='green')
# graphFrame.grid()

# # Create a label in the frame
# lmain = tk.Label(videoFrame)
# lmain.grid()

# # Controls
# controls = tk.Frame(app, bg='grey')
# controls.grid(row=1, column=0, columnspan=2, padx=20, pady=10)

root = tk.Tk()
root.minsize(1400, 750)
        # root.geometry('1400x750')

mainFrame = tk.Frame(root)
mainFrame.rowconfigure(0, weight=7)
mainFrame.rowconfigure(1, weight=1)
mainFrame.columnconfigure(0, weight=1)
mainFrame.columnconfigure(1, weight=12)
mainFrame.pack(fill=tk.BOTH, expand=True)

videoFrame = tk.Frame(mainFrame)
videoFrame.grid_propagate(False)
videoFrame.grid(row=0, column=0, sticky=tk.NSEW)

lmain = tk.Label(videoFrame)
lmain.pack_propagate(False)
lmain.pack()

graphFrame = tk.Frame(mainFrame, bg='green')
graphFrame.grid(row=0, column=1, sticky=tk.NSEW)


controlFrame = tk.Frame(mainFrame, bg='aliceblue')
controlFrame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, padx=30, pady=10)
controlFrame.columnconfigure(0, weight=1)
controlFrame.columnconfigure(1, weight=1)
controlFrame.columnconfigure(2, weight=1)

# **************************GUI****************************



# Capture from camera
detector = FaceMeshDetector(maxFaces=2)
cap = cv2.VideoCapture(0)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emo_freq_dict = {
    "Anger": 0,
    "Disgust": 0,
    "Fear": 0,
    "Happy": 0,
    "Neutral": 0,
    "Sad": 0,
    "Surprise": 0
} 
time_Count = 0
print('Outside video_stream()')
# function for video streaming
def video_stream():
    global time_Count
    global blinkCounter
    global counter
    global final_count
    global timed_blink
    global avg_blinks
    global lie_Counter
    global truth_Counter
    global neutral_Counter
    global y_EmoSum
    global color
    print('Inside video_stream()')
    start_timer = time.time()
    end_timer = time.time() + 60

    print('Inside if cap.get()')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        print('Inside if faces:')
        face = faces[0]
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and counter == 0:
            if (int(time.time() - start_timer) % 10 == 0):
                avg_blinks = round((timed_blink / 10) * 6)
                blinkList.append(avg_blinks)
                timed_blink = 1

                blinkCounter += 1
                final_count += 1
                color = (0, 200, 0)
                counter = 1
                # time.sleep(1)
            else:
                blinkCounter += 1
                final_count += 1
                timed_blink += 1
                color = (0, 200, 0)
                counter = 1

        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0


    _, frame = cap.read()
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    tk_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)
    

    for (x, y, w, h) in num_faces:
        serial = 1
        cv2.rectangle(tk_frame, (x, y-50), (x+w, y+h+10), (12, 181, 28), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        if maxindex == 0:  # Angry = -1
            y_EmoSum = + -1
            truth_Counter = + 1
            emo_freq_dict['Anger'] += 1

        elif maxindex == 1:  # Disgusted = 1
            y_EmoSum += 1
            lie_Counter += 1
            emo_freq_dict['Disgust'] += 1
                # Made every emotion 1 or -1 to make the graph more manageable
        elif maxindex == 2:  # Fear = 1
            y_EmoSum += 1
            lie_Counter += 1
            emo_freq_dict['Fear'] += 1

        elif maxindex == 3:  # Happy = -1
            y_EmoSum += -1
            truth_Counter += 1
            emo_freq_dict['Happy'] += 1

        elif maxindex == 4:  # Neutral = 0
            y_EmoSum = 0
            neutral_Counter += 1
            emo_freq_dict['Neutral'] += 1

        elif maxindex == 5:  # Sad = -1
            y_EmoSum += -1
            truth_Counter += 1
            emo_freq_dict['Sad'] += 1

        elif maxindex == 6:  # Surprised = 1
            y_EmoSum += 1
            lie_Counter += 1
            emo_freq_dict['Surprise'] += 1

        x_Time.append(time_Count)
        y_Emotion.append(y_EmoSum)

        dataframe = pd.DataFrame(
                list(zip(x_Time, y_Emotion)), columns=['Time', 'Emotions'])
        dataframe.to_csv("EmotionsDetected.csv")

        # print(maxindex)
        # print(final_count)

        print('Blink Counter: ', blinkCounter)

        time_Count = + 1
        cv2.putText(tk_frame, emotion_dict[maxindex], (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 35, 35), 2, cv2.LINE_AA)

    img = Image.fromarray(tk_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, video_stream) 

# # video_stream()
# start_button = tk.Button(controls, text="   Start   ", command=video_stream,fg="#ffffff", bg="#263942")
# # start_button.pack()
# stop_button = tk.Button(controls, text="   Stop   ",fg="#ffffff", bg="#263942")
# # stop_button.pack()
# end_button = tk.Button(controls, text="   End  ", command=lambda: print('ENDED'),fg="#ffffff", bg="#263942")
# # end_button.pack()

# start_button.grid(row=0, column=0, ipadx=20, ipady=10)
# stop_button.grid(row=0, column=1, ipadx=20, ipady=10)
# end_button.grid(row=0, column=2, ipadx=20, ipady=10)


startButton = tk.Button(controlFrame, height=2, width=12, text="   Start   ",command=video_stream, fg="#ffffff", bg="#263942", font=('Arial', 15))
startButton.grid(row=0, column=0, padx=10, pady=10, )
stopButton = tk.Button(controlFrame, height=2, width=12, text="   Stop   ",fg="#ffffff", bg="#263942", font=('Arial', 15))
stopButton.grid(row=0, column=1, padx=10, pady=10, )
endButton = tk.Button(controlFrame, height=2, width=12, text="   End  ", command=lambda: print('ENDED'),fg="#ffffff", bg="#263942", font=('Arial', 15))
endButton.grid(row=0, column=2, padx=10, pady=10, )


root.mainloop()