import cv2
import mediapipe as mp
import time
import pandas as pd

from pandas import DataFrame


mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_holistic=mp.solutions.holistic

count = 0
alldata =[]

pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

'''file load'''
cap = cv2.VideoCapture('dance.mp4')
background = cv2.VideoCapture('dance.mp4')

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    #posedata coordinate


    while cap.isOpened():
        success, image = cap.read()
        start = time.time()
        _,image=cap.read()
        _,backgroundimage=background.read()

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



        #print(result.pose_landmarks)

        #얼굴메쉬
        #mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        #왼손
       # mp_drawing.draw_landmarks(
       #     image,
       #     result.left_hand_landmarks,
       #     mp_holistic.HAND_CONNECTIONS)
            
        #오른손
       # mp_drawing.draw_landmarks(
       #     image,
       #     result.right_hand_landmarks,
       #     mp_holistic.HAND_CONNECTIONS)
      
        #바디 포즈
        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS)

        if result.pose_landmarks:
            data_tubuh = {}
            for i in range(len(pose_tangan)):
                result.pose_landmarks.landmark[i].x = result.pose_landmarks.landmark[i].x * image.shape[0]
                result.pose_landmarks.landmark[i].y = result.pose_landmarks.landmark[i].y * image.shape[1]
                data_tubuh.update(
                    {pose_tangan[i]: result.pose_landmarks.landmark[i]}
                )
                alldata.append(data_tubuh)

        #fps 계산
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)
        cv2.putText(image, f'FPS: {int(fps)}', (30,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),)


        cv2.imshow("zumba",image)
        key=cv2.waitKey(1)
        if key==ord('q'):
            df = pd.DataFrame(alldata)
            df.to_csv("dance_Coordinate.csv")
            print(alldata)
            break
cap.release()