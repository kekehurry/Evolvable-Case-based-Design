import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

def calculate(image,width=400,height=400,threshold=20,iterations=0):
    
    color_list = [(255,255,255),(68,58,130),(49,104,142),(33,144,141),(53,183,121),(143,215,68),(253,231,37)]
    level_list = [0,3,7,16,25,40,60]
    image = cv2.cvtColor(np.array(image,np.uint8), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,(width,height))
    

    footprints,areas,contours,heights,ids= [],[],[],[],[]
    for i in range(len(color_list)):
        r,g,b = color_list[i]
        level = level_list[i]
        color = np.uint8([int(b),int(g),int(r)])
        low_b = color-threshold
        up_b = color+threshold
        if i==0 : up_b= np.uint8([255,255,255])
        thresh = cv2.inRange(image,low_b,up_b)
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=iterations)
        co, _ = cv2.findContours(thresh , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in co:
            footprint = cv2.contourArea(c)
            if footprint < 20:
                continue
            area = footprint*level
            footprints.append(footprint) 
            areas.append(area) 
            contours.append(c)
            heights.append(level*3)
            ids.append(i)
            
    FSI= np.sum(areas)/(width*height)
    GSI = np.sum(footprints)/(width*height)
    L = FSI/GSI
    OSR = (1-GSI)/FSI
    data = (FSI,GSI,L,OSR)
    return data,contours,heights,ids

if __name__ == '__main__':
    # FILE = r'datasets\Shenzhen\train\images\id_1512.png'
    # image = Image.open(FILE)
    # data,contours,heights,ids = calculate(image)
    # print(data)


    folder = r'datasets\Shenzhen\train\images'
    df = pd.DataFrame(columns=['FILE','FSI','GSI','L','OSR'])
    for i,file in enumerate(os.listdir(folder)):
        if file.endswith('.png'):
            filepath = os.path.join(folder,file)
            img = Image.open(filepath).convert('RGB')
            data,contours,heights,ids = calculate(img)
            df.loc[i] = [filepath,data[0],data[1],data[2],data[3]]
            print(f'processing {i}/{len(os.listdir(folder))}...')
    
    df.to_excel('styles/data.xlsx')


