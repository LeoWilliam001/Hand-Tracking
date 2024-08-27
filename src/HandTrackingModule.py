import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands=mp.solutions.hands
        # self.hands=self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.hands.process(imgRGB)

    # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for screen in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, screen, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):

        lmList=[]

        if self.result.multi_hand_landmarks:
            myHand=self.result.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                height,width,channels=img.shape
                # The height ranges from 480 to 640
                print(id,width,height)
                # lm.x and lm.y provides the landmark where 0.5 is present in the moddle
                print(id,lm.x,lm.y)
                # The exact coordinates are calculated in this particular line of code
                cx,cy=int(lm.x*width),int(lm.y*height)
                lmList.append([id,cx,cy])
                print(id,cx,cy)
                # if(id==8):
                #     cv2.circle(img,(cx,cy),25,(255,0,0),cv2.FILLED)
        return lmList


def main():
    cap=cv2.VideoCapture(0)
    detector=handDetector()
    cTime=0
    pTime=0

    while True:
        success, img=cap.read()
        img = cv2.flip(img, 1)
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[8])

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()
