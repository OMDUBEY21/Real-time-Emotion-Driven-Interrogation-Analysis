 <div align="center">

# Real-time Emotion-Driven Interrogation Analysis 

</div>

## CrimiFace: Real-time Emotion-Driven Interrogation Analysis

### Introduction:
Facial Analysis for Suspect Interrogation is a real-time machine learning-based application used for detecting suspect with the help of computer vision.

### Problem statement:
To elicit confessions from criminal suspects, police officers employ a range of strategies. Law
enforcement observes a suspect’s verbal and non-verbal cues during an interrogation. The key
areas on which attention is concentrated during interrogation are facial expression and body
language. For law enforcement to assess whether the suspect is telling the truth or not, the
interrogation contains behavior-provoking questions. However, despite all of their training,
cops may still have some questions about whether a particular person has committed a crime
or not. Therefore, an emotion facial analysis algorithm might be of great help to officers in providing
them with a general impression of the suspect. During each interrogation, a camera will record
all of the suspect’s facial expressions and eye blinks. Using an algorithm, these expressions will be transformed
into a graphical representation, and the investigator will examine the most prevalent expressions
in this frame of the graph to reach a preliminary determination of whether the suspect is telling
the truth or not which will furthere generate a report of overall session investigation and helps the
investigator to conclude whether the suspect is criminal or not.

Used Tech Stack
1. Keras
2. Tensorflow
3. Numpy
4. Matplotlib
5. Tkinter
6. PIL
7. Opencv
8. Mongodb

#### Install

1. Create a virtual environment

    `virtualenv venv`

    Or

    `python3.8 -m venv venv`

2. Activate it

    `source venv/Scripte/activate`

3. Clone the repository and install the packages in the virtual env:

    `pip install -r requirements.txt`

4. Download and install Mongodb Compass:
       Create Database EmDetect & Collection as reports
    
GUI:

![p1](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/9d1c2e01-59f2-43dc-a702-fcf5ca6fcd19)
<br>
<br>
Happy: The graph displaying a happy face indicates that the suspect is expressing happiness when answering questions related to the crime during the interrogation. This suggests that the suspect may be telling the truth that he/she is not having any involvement in the crime.
![p2](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/ef7e06d8-c66b-41fb-8be5-8af020405231)
<br>
<br>
Fearful: The graph displays a fearful facial expression exhibited by the suspect when asked questions related to the crime during the interrogation. This expression suggests that the suspect may be lying or experiencing fear and anxiety in response to those specific questions.
![p3](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/eeac3304-bea1-4cf7-9e46-73cedc5bcae6)
<br>
<br>
Surprised: Here, The presence of a surprising expression on the graph during the interrogation period indicates that the suspect is visibly surprised by a provocative question, and the graph accurately reflects the suspect's real-time emotional response. 
![p5](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/ef772395-1f52-4ce1-942d-6618109884ed)
<br>
<br>
Overall: Now, This is the overall graph generated representing the cumulative facial expressions of the suspect throughout the entire interrogation session, specifically when asked questions pertaining to the crime. 
![overall](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/7c7f1a7d-4ba4-4c5f-9127-d755eb924527)


Report:
<br>
At the conclusion of the interrogation, a final PDF file is generated that provides the interrogator with valuable insights regarding the potential involvement of the suspect in the crime. This PDF report includes the collected data from each interrogation session, which is stored in the backend database i.e MongoDB. The information presented in the report aids the interrogator in making informed decisions and forming rough conclusions about the suspect's culpability based on the data analysis and assessment of their responses throughout the interrogation process.
<br>
<br>

   Session 1 Report:
   
   ![l1](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/fb579434-cf33-463a-8d6b-23caf353e059)
   <br>
   <br>
   Backend: Database of session 1
   
   ![l1r](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/306af9db-ea75-484b-97d7-32b1632399ab)
    <br>
    <br>
    Session 3 Report:
    
   ![l2](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/7467c58a-cff8-42c5-83d6-df5cedd959a2)
   <br>
   <br>
   Backend: Database of session 3
   
   ![l2r](https://github.com/OMDUBEY21/Facial_Analysis_for_Suspect_Interrogation/assets/84987833/7ad159aa-fd40-4ff9-9007-810a50f51fc1)    

    
