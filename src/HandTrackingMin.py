import cv2
import mediapipe as mp
import time

#To open output as video screen
cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

#Initializing the current time and previous time
cTime=0
pTime=0

while True:
    #success returns true if frame is available
    success, img=cap.read()
    #cv2 reads the image as BGR view which needs to be converted into RGB view in order to be processed
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRGB)
    # print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for screen in result.multi_hand_landmarks:
            for id,lm in enumerate(screen.landmark):
                height,width,channels=img.shape
                # The height ranges from 480 to 640
                print(id,width,height)
                # lm.x and lm.y provides the landmark where 0.5 is present in the moddle
                print(id,lm.x,lm.y)
                # The exact coordinates are calculated in this particular line of code
                cx,cy=int(lm.x*width),int(lm.y*height)
                print(id,cx,cy)
                if(id==8):
                    #cv2.circle(image, center_coordinates, radius, color,thickness )
                    cv2.circle(img,(cx,cy),25,(255,0,0),cv2.FILLED)
            #mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(img, screen, mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    #cv2.putText(image, text, org, font, fontScale, color, thickness, lineType)
    cv2.putText(img,str(int(fps)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
