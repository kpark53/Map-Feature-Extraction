import cv2
import numpy as np


def getROICnts(ROI):
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    erode = cv2.erode(thresh, kernel, iterations=3)
    dilate = cv2.dilate(erode,kernel,iterations=5)
    edges = cv2.Canny(dilate,100,230)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return cnts


class FeatMatcher:
    
    def __init__(self,FLANN_params={}):
        search_params = FLANN_params.get("search",{})
        index_params = FLANN_params.get("index",dict(algorithm=0,trees = 5))
        self.g_match_thresh = FLANN_params.get("g_match_thresh",0.85)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.featExtractor = cv2.SIFT_create()
        
    def computeFeats(self,ROI):
        img_kp, img_des = self.featExtractor.detectAndCompute(ROI, None)
        return img_kp,img_des
    
    def getMatches(self,img_des,template_des,img_kp,k=2,top=90):
        matches = self.matcher.knnMatch(template_des, img_des, k=k)
        good_matches = []
        for match in matches:
            good_matches.extend([[m.trainIdx,m.distance] for m in match])
        good_matches = np.array(sorted(good_matches,key=lambda x : x[1]))[:top]
        if(len(good_matches)==0):
            return [],[]
        #print(good_matches.shape)
        good_matches = good_matches[:,0].astype(int)
        src_points = np.unique(np.int32([img_kp[m].pt for m in good_matches]).reshape(-1, 2),axis=0)
        return src_points,matches


def getNameVectDict(names):
    vect_dict = {}
    for i, name in enumerate(names):
        #label = shape
        name = name.lower()
        for char in name:
            vect_dict[char] = vect_dict.get(char,0) + 1
    for i,key in enumerate(vect_dict.keys()):
        vect_dict[key] = i
    num_chars = len(vect_dict)
    name_vect = np.zeros((len(names),num_chars))
    for i, name in enumerate(names):
        name = name.lower()
        for char in name:
            name_vect[i,vect_dict[char]] += 1
    return name_vect,vect_dict


def getVectDists(text,name_vect,vect_dict):
    text_vect = np.zeros(len(vect_dict))
    valid_chars = vect_dict.keys()
    text = text.lower()
    text = "".join([char for char in text if char in valid_chars])
    for char in text:
        text_vect[vect_dict[char]] += 1
    return np.sum(name_vect-text_vect,axis=1)**2



def colorFilters(ref_colors,centroid_color,candidates,thresh = 20):
    min_diff = 255
    out = 0
    for i in candidates:
        diff = np.sqrt(np.sum((ref_colors[i]-centroid_color)**2))
        if(diff<thresh):
            return i,1
        if(diff<min_diff):
            min_diff = diff
            out = i
    return out,0