# Importing Libraries
import cv2
import time
#import numpy as np
import matplotlib.pyplot as plt
from utils.geometric import cropImage, getCloseCnts, superpixel, withinMargin
from utils.dataReader import getNames, loadJson, getTemplateCoords
from utils.detectors import getROICnts, FeatMatcher, getNameVectDict, getVectDists
from utils.ocr import ocr
import cupy as np
import torch




# Defining Constants/Parameters
DATAPATH = "C:/code/"
MARGIN = 10
MAX_DIST = 4
MIN_DIST = 2

# Loading Path Names
names = getNames(DATAPATH)

# Initializing Variables
matcher = FeatMatcher()

# Detection Loop
for name in names[2:4]:
    start = time.time()
    img = cv2.imread(name+".tif")
    ROI,crop_coords = cropImage(img)
   # coord = ocr(ROI)
    ROI_cnts = getROICnts(ROI)
    plt.imshow(ROI)
    plt.show()
    json_data = loadJson(name)
    img_kp, img_des = matcher.computeFeats(ROI)
    print("Computed Image Features")
    #Superpixel
    segments,props = superpixel(ROI)
    print("Computed Superpixels")
    poly_shapes = ["".join(shape["label"].split("_")[:-1]) for shape in json_data["shapes"] if shape["label"].split("_")[-1]=="poly"]
    name_vect,vect_dict = getNameVectDict(poly_shapes)
    centroids = [[int(c) for c in prop.centroid] for prop in props]
    prop_labels = [prop.label for prop in props]
    #name_vect, vect_dict = getNameVectDict(poly_shapes)
    for legend in json_data["shapes"]:
        label = legend["label"]
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
            match_points,matches = matcher.getMatches(img_des,template_des,img_kp,k=5,top=100)
            blank = np.zeros(ROI.shape).astype(np.uint8)
            blank_re=np.asnumpy(blank)
            cnts = []
            for point in match_points:
                cnts.extend(getCloseCnts((point[1],point[0]),ROI_cnts,20))
            if(len(cnts)!=0):
                cv2.drawContours(blank_re,cnts,-1,(255,255,255),3)
            plt.imshow(blank_re)
            plt.show()
        elif(typ=="pt"):
            plt.imshow(template)
            plt.show()
            template_kp, template_des = matcher.computeFeats(template)
            match_points,matches = matcher.getMatches(img_des,template_des,img_kp,k=5)
            blank = ROI.copy()
            for point in match_points:
                cv2.circle(blank, (point[1],point[0]), 50, (255,0,0), -1)
            plt.imshow(blank)
            plt.show()
        else: # Points
            #getVectDists(text,name_vect, vect_dict)
            plt.imshow(template)
            plt.show()
            r=int(np.median(np.array(template[:,:,0])))
            g=int(np.median(np.array(template[:,:,1])))
            b=int(np.median(np.array(template[:,:,2])))
            candidates = []
            mask = np.zeros(ROI.shape[:2])
            for i,centroid in enumerate(centroids):
                if(withinMargin((r,g,b),ROI[centroid[0],centroid[1]],MARGIN)):
                    if((ROI[centroid[0],centroid[1]]>=(240,240,240)).all() or (ROI[centroid[0],centroid[1]]<=(10,10,10)).all()):
                        continue
                    candidates.append([centroid,prop_labels[i]])
                    mask_add = (np.equal(np.array(segments),np.array(prop_labels[i])))
                    mask = mask + mask_add
            mask = mask>0
            text_img = np.array(ROI) * np.array(mask.reshape((*mask.shape,1)))
            text_img = np.asnumpy(text_img.get())
            text_coords = ocr(text_img)
            mask_re = np.asnumpy(mask.get())
            
            for text, coord in text_coords:
                print(text)
                dist = getVectDists(text,name_vect,vect_dict)[poly_shapes.index(name)]
                if(dist>MAX_DIST):
                    #Check if x and y are at the correct positions
                    curr_label = segments[int(coord[0][1]),int(coord[0][0])] 
                    mask = np.array(mask) * (np.not_equal(np.array(segments),np.array(curr_label)))
                    mask_re = np.asnumpy(mask.get())
            plt.imshow(mask_re)
            plt.show()
        
        
    end = time.time()
    print(f"Total Time Elapsed : {end-start}")
