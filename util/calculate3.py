import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

def calculate(image,width=400,height=400,threshold=50,iterations=0):
    
    color_list = [(255,255,255),(68,58,130),(49,104,142),(33,144,141),(53,183,121),(143,215,68),(253,231,37)]
    level_list = [0,3,7,16,25,40,60]
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,(width,height))

    footprints,areas,contours,heights,ids= [],[],[],[],[]
    for i in range(len(color_list)):
        r,g,b = color_list[i]
        level = level_list[i]
        color = np.uint8([int(b),int(g),int(r)])
        low_b = color-threshold
        up_b = color+threshold
        if i==0 : up_b= np.uint8([255,255,255]) 
        map = cv2.inRange(image,low_b,up_b) 
        _ , thresh = cv2.threshold(map,127,255,cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=iterations)
        co, _ = cv2.findContours(thresh , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for c in co:
            footprint = cv2.contourArea(c)
            if footprint < 10:
                continue
            if level !=0:
                area = footprint*level
                footprints.append(footprint) 
                areas.append(area) 
            contours.append(c)
            heights.append(level*3)
            ids.append(i)
            

    BCR = np.sum(footprints)/(width*height)
    FAR = np.sum(areas)/(width*height)

    data = (BCR,FAR)

    return data,contours,heights,ids

if __name__ == '__main__':
    
    # dir = r'datasets\shenzhen\train\1'
    # file_list = [os.path.join(dir,x) for x in os.listdir(dir) if x[-3:]=='png']
    # df = pd.DataFrame(columns=['FILE','BCR','FAR','KD'])
   
    # for i  in tqdm(range(len(file_list))):
    #     FILE = file_list[i]
    #     image = cv2.imread(FILE)
    #     data,contours,heights = calculate(image)
    #     BCR,FAR,KD = data
    #     df.loc[i] =  dict(zip(df.columns, [FILE,BCR,FAR,KD]))
    
    # df.to_excel('analysis/train_data2.xlsx',index=True)

    from PIL import Image
    FILE = r'id_1383.png'
    image = Image.open(FILE)
    data,contours,heights = calculate(image)
    print(contours)
