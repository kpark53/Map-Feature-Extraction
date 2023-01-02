import os
import cv2
import easyocr
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True' #avoid initializaing libiomp5md.dll error
"""
self, lang_list, gpu=True, model_storage_directory=None,
                 user_network_directory=None, detect_network="craft", 
                 recog_network='standard', download_enabled=True, 
                 detector=True, recognizer=True, verbose=True, 
                 quantize=True, cudnn_benchmark=False
"""

reader = easyocr.Reader(['en'],recog_network='english_g2')

def ocr(ROI):
    '''
    text_coord : list
    [text, coordiates1, 2, 3, 4 ]
    '''
    ROI = ROI.astype(np.float32)
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text_coord=[] 
    bounds = reader.readtext(ROI)
    for i in range(len(bounds)):
        text= bounds[i][1]
        coord= bounds[i][0]
        text_coord.append((text,coord))  #dictionary with recognized text and coordinate from OCR
    return text_coord