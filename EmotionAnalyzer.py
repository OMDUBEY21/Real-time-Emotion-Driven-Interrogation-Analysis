import cv2
import csv
import time
import pymongo
import threading
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from utils import Report, Session
from FunctionDB import create_session
from pdf_generator import generate_pdf
from keras.models import model_from_json
from matplotlib.animation import FuncAnimation
from cvzone.FaceMeshModule import FaceMeshDetector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

client = pymongo.MongoClient("mongodb://localhost:27017/")
collection = client.EmDetect.reports

TEST_oid = '6482cb09eb9ea96a791962f8'


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
detector = FaceMeshDetector(maxFaces=2)

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


# Create a window with a label to display the video
# ****************************** GUI *******************************
window = tk.Tk()
window.minsize(1400, 750)
# label = tk.Label(window)
# label.pack()

mainFrame = tk.Frame(window)
mainFrame.rowconfigure(0, weight=7)
mainFrame.rowconfigure(1, weight=1)
mainFrame.columnconfigure(0, weight=1)
mainFrame.columnconfigure(1, weight=15)
mainFrame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

videoFrame = tk.Frame(mainFrame)
videoFrame.grid_propagate(False)
videoFrame.grid(row=0, column=0, sticky=tk.E+tk.W)

label = tk.Label(videoFrame, padx=0, pady=0, borderwidth=0, highlightthickness=0)
label.pack_propagate(False)
label.pack()

graphFrame = tk.Frame(mainFrame)
graphFrame.grid(row=0, column=1, sticky=tk.E+tk.W)
print('graphFrame created')
fig, ax = plt.subplots(figsize=(8,5))
canvas = FigureCanvasTkAgg(fig, graphFrame)
canvas.get_tk_widget().pack()
print('canvas created')


controlFrame = tk.Frame(mainFrame)
controlFrame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, padx=30, pady=10)
controlFrame.columnconfigure(0, weight=1)
controlFrame.columnconfigure(1, weight=1)
controlFrame.columnconfigure(2, weight=1)

# ****************************** GUI *******************************
print('GUI created')

# record_flag = False
stop_event = threading.Event()

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

def graph_plot():
    while not stop_event.is_set():
        df = pd.read_csv('EmotionsDetected.csv')
        x = df['Time']
        y = df['Emotions']
        ax.clear()
        ax.plot(y, label="Emotion", lw =1, color = 'Red')
        ax.set_ylim(-30, 30)
        ax.set_yticks((-15, 0, 15), ('Truth', 'Neutral', 'Lies'), color = 'Green')
        plt.legend(loc = 'upper left')
        plt.xlabel("Time", color = 'Blue')
        plt.tight_layout()
        canvas.draw()
        time.sleep(0.01)

def update_video():
    global stop_event
    while not stop_event.is_set():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        photo = ImageTk.PhotoImage(image)

        label.config(image=photo)
        label.image = photo

        time.sleep(0.01)


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
    start_timer = time.time()
    end_timer = time.time() + 60

    while not stop_event.is_set():
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
        img =  img.resize((750, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        print(img.size)
        label.configure(image=imgtk)
        label.image = imgtk

        time.sleep(0.01)

    with open('ReportData.csv', 'a', newline='') as csvfile:
        csvwrite = csv.writer(csvfile)
        csvwrite.writerow(['Emotion Frequency', 'Blink List', 'Truth Count', 'Lie Count', 'Neutral Count'])
        csvwrite.writerow([emo_freq_dict, blinkList, truth_Counter, lie_Counter, neutral_Counter])
    
    # print("Lied Percentage : ", int((lie_Counter/(lie_Counter + truth_Counter + neutral_Counter))*100), " ", type(lie_Counter))
    # print("Truth Percentage : ", int((truth_Counter/(lie_Counter + truth_Counter + neutral_Counter))*100), " ", type(truth_Counter))
    # print("Neutral Percentage : ", int((neutral_Counter/(lie_Counter + truth_Counter + neutral_Counter))*100), " ", type(neutral_Counter))
    # print("\nBase Blink rate  : ", blinkList[0])
    # print("Highest Blink Rate : ", max(blinkList))
    # print("Average Blink List : ", sum(blinkList)/len(blinkList))
    # print("Blink list : ", blinkList, " ", type(blinkList))
    # print("\n")
    # print(emo_freq_dict, " ", type(emo_freq_dict))

    lie_percent = int((lie_Counter/(lie_Counter + truth_Counter + neutral_Counter))*100)
    truth_percent = int((truth_Counter/(lie_Counter + truth_Counter + neutral_Counter))*100)
    neutral_percent = int((neutral_Counter/(lie_Counter + truth_Counter + neutral_Counter))*100)
    base_blink = blinkList[0]
    max_blink = max(blinkList)
    avg_blink = sum(blinkList)/len(blinkList)
    blink_stat = blinkList
    emotion_stat = emo_freq_dict

    session_data = Session(
        blink_stat={
            'base': base_blink,
            'avg': avg_blink,
            'max': max_blink
        }, 
        emotion_stat=emotion_stat,
        truth_percent=truth_percent, 
        lie_percent=lie_percent,
        neutral_percent=neutral_percent
    )
    print('created session data')
    create_session(session=session_data, oid=TEST_oid)
    print('inserted session data')
    

def start_thread():
    global video_thread, stop_event
    stop_event.clear()
    video_thread = threading.Thread(target=video_stream)
    graph_thread = threading.Thread(target=graph_plot)
    video_thread.start()
    time.sleep(5)
    graph_thread.start()

def stop_thread():
    global stop_event, label
    stop_event.set()
    white_back = cv2.imread('white_back.png')
    label.config(image=white_back)
    label.image = white_back

def pdf_generator_callback():
    generate_pdf(TEST_oid)


startButton = tk.Button(controlFrame, height=2, width=12, text="   Start   ", command=start_thread, fg="#ffffff", bg="#263942", font=('Arial', 15))
startButton.grid(row=0, column=0, padx=10, pady=10, )
stopButton = tk.Button(controlFrame, height=2, width=12, text="   Stop   ", command=stop_thread, fg="#ffffff", bg="#263942", font=('Arial', 15))
stopButton.grid(row=0, column=1, padx=10, pady=10, )
endButton = tk.Button(controlFrame, height=2, width=12, text="   End  ", command=pdf_generator_callback, fg="#ffffff", bg="#263942", font=('Arial', 15))
endButton.grid(row=0, column=2, padx=10, pady=10, )

window.mainloop()

cap.release()
cv2.destroyAllWindows()
