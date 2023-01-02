# Importing Libraries
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from utils.geometric import cropImage, getCloseCnts, superpixel, getMasked, checkNeigbors
from utils.dataReader import getNames, loadJson, getTemplateCoords
from utils.detectors import getROICnts, FeatMatcher, getNameVectDict, getVectDists
from utils.ocr import ocr
#
# Defining Constants/Parameters
#DATAPATH = "D:/Datasets/AICMA/training/"
DATAPATH = "C:/ai4cma/"
OUTPATH = "C:/darpa/outputs_second/"
MARGIN = 10
MAX_DIST = 2
DOWN_WEIGHT = -0.05
UP_WEIGHT = 1
SEARCH_SIZE = (21,21)

# Loading Path Names
names = getNames(DATAPATH)

# Initializing Variables
matcher = FeatMatcher()

# Initalizaing Saving Dir
if(not(os.path.isdir(OUTPATH))):
    os.mkdir(OUTPATH)
    
# Detection Loop
for fname in names:
    start = time.time()
    img = cv2.imread(fname+".tif")
    ROI,crop_coords = cropImage(img,fname)
    ROI_hue = rgb2hsv(ROI)[:,:,0]
   # coord = ocr(ROI)
    ROI_cnts = getROICnts(ROI)
    plt.imshow(ROI)
    plt.show()
    json_data = loadJson(fname)
    img_kp, img_des = matcher.computeFeats(ROI)
    print("Computed Image Features")
    #Superpixel
    segments,props = superpixel(ROI)
    print("Computed Superpixels")
    poly_shapes = ["".join(shape["label"].split("_")[:-1]) for shape in json_data["shapes"] if shape["label"].split("_")[-1]=="poly"]
    name_vect,vect_dict = getNameVectDict(poly_shapes)
    centroids = [[int(c) for c in prop.centroid] for prop in props]
    prop_labels = [prop.label for prop in props]
    text_coords = ocr(ROI)
    print(",".join([text for text,coord in text_coords]))
    curr_labels = set()
    
    #name_vect, vect_dict = getNameVectDict(poly_shapes)
    for legend in json_data["shapes"]:
        label = legend["label"]
        if(label in curr_labels):
            continue
        else:
            curr_labels.add(label)

        outname = OUTPATH + fname.split("/")[-1] + "_" +label + ".tif"
        name = "".join(label.split("_")[:-1])
        typ = label.split("_")[-1]
        y_min,y_max,x_min,x_max = getTemplateCoords(legend)
        template = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        h_t, w_t = template.shape[0], template.shape[1]
        ref_labels = {}
        unique_colors = set()
        if(typ=="line"):
            plt.imshow(template)
            plt.show()
            template_kp, template_des = matcher.computeFeats(template)
            match_points,matches = matcher.getMatches(img_des,template_des,img_kp,k=5,top=200)
            blank = np.zeros(ROI.shape).astype(np.uint8)
            cnts = []
            for point in match_points:
                cnts.extend(getCloseCnts((point[1],point[0]),ROI_cnts,20))
            if(len(cnts)!=0):
                cv2.drawContours(blank,cnts,-1,(255,255,255),3)
        elif(typ=="pt" or typ == "point"):
            plt.imshow(template)
            plt.show()
            template_kp, template_des = matcher.computeFeats(template)
            match_points,matches = matcher.getMatches(img_des,template_des,img_kp,k=5)
            blank = np.zeros(ROI.shape).astype(np.uint8)
            for point in match_points:
                blank[point[1],point[0]] = 255
        else: # Polygons
            plt.imshow(template)
            plt.show()
            r=int(np.median(template[:,:,0]))
            g=int(np.median(template[:,:,1]))
            b=int(np.median(template[:,:,2]))
            candidates = []
            curr_index = poly_shapes.index(name)
            mask = np.zeros(ROI.shape[:2])
            mask = getMasked(ROI,(r,g,b),MARGIN)>0
            for text, coord in text_coords:
                x1,y1 = coord[0]
                if(not(checkNeigbors((r,g,b),ROI,(y1,x1),SEARCH_SIZE,MARGIN))):
                    continue
                dists = getVectDists(text,name_vect,vect_dict)
                min_index = np.argmin(dists)
                dist = dists[min_index]
                if(dist<MAX_DIST and min_index!=curr_index):
                    curr_label = segments[y1,x1] 
                    mask = mask + DOWN_WEIGHT * (segments==curr_label)
                elif(dist<MAX_DIST and min_index==curr_index):
                    curr_label = segments[(y1,x1)] 
                    mask = mask + UP_WEIGHT * (segments==curr_label)
            m = (mask>0).astype(np.uint8)*255
            blank = np.zeros(ROI.shape).astype(np.uint8)
            blank[:,:,0]=m
            blank[:,:,1]=m
            blank[:,:,2]=m
        out = np.zeros(img.shape).astype(np.uint8)
        x,y,w,h = crop_coords
        out[y:y+h,x:x+w] = blank
        cv2.imwrite(outname,out)
    end = time.time()
    print(f"Total Time Elapsed : {end-start}")
