import cv2
import numpy as np

class FaceDetections:
    
    def __init__(self, frame, sizes):
        self.frame = frame
        self.HaarCascadeFace = cv2.CascadeClassifier('CasCade/Face_Detect.xml')
        self.size = sizes
    
    # Crop Face from images
    def CropFaces(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        face_rec = self.HaarCascadeFace.detectMultiScale(self.frame, scaleFactor=1.2, minNeighbors=5)
        if len(face_rec)!=0:
            for (x,y,w,h) in face_rec:
                self.faces_detec = self.frame[y-25:h+y+25, x-25:x+w+25]
                self.faces_detec = cv2.resize(self.faces_detec, self.size).astype(np.float32)
        return self.faces_detec
    
    # If the number of data is small, we need to increase it 
    def AugmentationImage(self):
        self.rotated_frame = cv2.rotate(self.faces_detec, cv2.ROTATE_90_CLOCKWISE)
        self.flipped_frame = cv2.flip(self.faces_detec, 1)
        kernel = np.array([[-1,-1,-1], 
                           [-1, 9,-1],
                           [-1,-1,-1]])
        self.sharpened_frame = cv2.filter2D(self.faces_detec, -1, kernel)
        
        return self.rotated_frame, self.flipped_frame, self.sharpened_frame
    
    def ConvertToRGB(self):
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
    
    def Predictions(self, predict_model, names_access):
        face_rec = self.HaarCascadeFace.detectMultiScale(self.frame, scaleFactor=1.2, minNeighbors=5)
        if len(face_rec)!=0:
            for (x,y,w,h) in face_rec:
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 255), 4)
                cv2.putText(self.frame, names_access, (y, y+h), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                
        return self.frame
                
                
