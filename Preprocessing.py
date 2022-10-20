import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ImageProcessing import FaceDetections

SIZE = (85, 85) # resize images

PATH_PERSONS_READ = 'Amir' # target face person
PATH_PERSONS_WRRITE = 'pos'# target face person
PATH_ETC_READ = 'ETC' # how many persons's face except target face
PATH_ETC_WRRITE = 'neg' # how many persons's face except target face

def ReadImage(path, path_to):
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        img = cv2.imread(image_path)
        crop_face = FaceDetections(img, SIZE)
        try:
            img = crop_face.CropFaces()
            if img is not None:
                rotated_frame, flipped_frame, sharpened_frame = crop_face.AugmentationImage()
                image_name_rotate = image_name[:-4]+'1'
                image_name_flipped = image_name[:-4]+'2'
                image_name_sharpened = image_name[:-4]+'3'
                cv2.imwrite(f'{path_to}/{image_name[:-4]}.png', img)
                cv2.imwrite(f'{path_to}/{image_name_rotate}.png', rotated_frame)
                cv2.imwrite(f'{path_to}/{image_name_flipped}.png', flipped_frame)
                cv2.imwrite(f'{path_to}/{image_name_sharpened}.png', sharpened_frame)
                print(f"this image {image_name[:-4]} done...")
                cv2.waitKey(1)
        except:
            print(image_name[:-4])
    
    
ReadImage(PATH_PERSONS_READ, PATH_PERSONS_WRRITE)
ReadImage(PATH_ETC_READ, PATH_ETC_WRRITE)
