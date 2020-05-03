import cv2
from GazeTracking.gaze_tracking import GazeTracking
import dlib
import os
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import os
import time

c = 0

class VideoDetector():
    
    def __init__(self, detection_model_path, emotion_model_path, img_dir=None):
        self.gaze = GazeTracking()
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.emotions = ("angry" ,"disgust", "scared", "happy", "sad", "surprised", "neutral")
        self.img_dir = img_dir

    def extract_images(self, video_path, freq=5):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        
        count = 0
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print(video_length)
        
        while count < video_length:
            success, image = vidcap.read()
            print (count, success)
            if success and count % freq == 0:
                cv2.imwrite(self.img_dir + "\\frame%d.jpg" % count, image)
            
            count += 1   
        return True
    
    def _make_img(self, f):  
        global c
        self.gaze.refresh(f)
        new_frame = self.gaze.annotated_frame()
            
        text, coord_left, coord_right = self._detect_gaze()
        emo_texts, max_ind = self._detect_emotions(f)
            
        cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        for line, y in zip((coord_left, coord_right), (100, 130)):
            print(line)
            cv2.putText(new_frame, line, (60, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
        for i in range(len(emo_texts)):
            color = (0,0,0)
                
            if i == max_ind:
                color = (0,0,255)
                    
            cv2.putText(new_frame, emo_texts[i], (60, 170+30*i), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2)
            print(emo_texts[i])
            
        cv2.imshow("annotated", new_frame)
        
    def annotate_imgs(self):
        if self.img_dir:
            for f in os.listdir(self.img_dir):
                print(f)
                im = os.path.join(self.img_dir, frame)
                f = cv2.imread(im)
                self._make_img(f)
                if cv2.waitKey(1) == 27:
                    break
        else:
            #cv2.namedWindow("online")
            vc = cv2.VideoCapture(0)
            if vc.isOpened(): # try to get the first frame
                rval, frame = vc.read()
            else:
                rval = False
                
            while rval:    
                self._make_img(frame)
                rval, frame = vc.read()
                #time.sleep(1)
                if cv2.waitKey(1) == 27:
                    break      
        vc.release()
        cv2.destroyAllWindows()
        
    def _detect_gaze(self):
        text = ""
        if self.gaze.is_right():
            text = "Looking right"
        elif self.gaze.is_left():
            text = "Looking left"
        elif self.gaze.is_center():
            text = "Looking center"

        coord_left = f'left eye: {self.gaze.pupil_left_coords()}'
        coord_right = f'right eye: {self.gaze.pupil_right_coords()}'
        #ratio_ver = f'vertical ratio: {round(float(self.gaze.vertical_ratio()),3)}'
        #ratio_hor = f'horizontal ratio: {round(float(self.gaze.horizontal_ratio()),3)}'
    
        return (text, coord_left, coord_right)
    

    def _detect_emotions(self, frame):
        emo_frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(emo_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

        emo_texts = []
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = self.emotions[preds.argmax()]

            max_ind = np.argmax(preds)
            for (i, (emotion, prob)) in enumerate(zip(self.emotions, preds)):
                text = f'{emotion}, {round(prob*100,3)}'
                emo_texts.append(text)   
        else:
            emo_texts = [f'{emotion} ???' for emotion in self.emotions] 
            max_ind = None
            
        return (emo_texts, max_ind)
    

           
