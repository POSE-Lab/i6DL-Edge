import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
import cv2 as cv 

def plot_pose_mask(pred_obj_label,original_image):

    image = Image.fromarray(pred_obj_label)
    image.convert("L")
    image = image.resize((1280,720),resample=Image.Resampling.BICUBIC)
    image = np.reshape(image,(720,-1,1))
    #cv.imshow("test",np.array(image))
    print(image.shape)
    #maskc = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

    # Taking a matrix of size 5 as the kernel
    kernel2 = np.ones((10, 10), np.float32)/100
    
    # Applying the filter
    image = cv.filter2D(src=image, ddepth=-1, kernel=kernel2)
  
    
    # contours = cv.findContours(image, 
    #     cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]


    # image_copy = image.copy()
    # cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, 
    #              color=(255, 255, 255), thickness=2, lineType=cv.LINE_AA)

    # x,y,w,h = cv.boundingRect(contours[0])
    # print("x,y,w,h",x,y,w,h)
    # cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255) , 1)

    cv.imshow('test',image)
    cv.waitKey(1000)

    image = Image.fromarray(image)

    image.putalpha(127)
    original_image.paste(image, (0, 0), image)
    original_image.show()

def plot_examples(image,conf,savePath):
    """
    Helper function to plot data with associated colormap.
    """
    #colormaps = [ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])]


    # colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
    # my_cmap = [ListedColormap(colors, name="my_cmap")]

    viridis = mpl.colormaps['viridis'].resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    newcmp = [ListedColormap(newcolors)]

    
    fig, axs = plt.subplots(1, 1, figsize=(1 * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, newcmp):
        psm = ax.pcolormesh(image, cmap=cmap, rasterized=True, vmin=0, vmax=255)
        fig.colorbar(psm, ax=ax)
    plt.gca().invert_yaxis()
    bbox = dict(boxstyle ="round", fc ="1.0")
    plt.annotate("Avg conf: "+str(np.round(conf,3)),(95,110),bbox=bbox)
    plt.savefig(savePath)
    #plt.show()

def confidense_calc(image,model_id):

    #get indices of pixels were the object is present
    # get confidense for every pixel
    # sum all the confidensed and compute average

    max_pixels = np.argmax(image,axis=-1)
    if len(np.where(max_pixels == model_id))!=0:
        obj_pixels = list(zip(*np.where(max_pixels == model_id)))
        #print(obj_pixels)
        conf_add = 0
        for i,j in obj_pixels:
            conf = image[i][j][model_id]
            conf_add+=conf
            #print(conf)
    else:
        return -999,-999
    if len(obj_pixels)!=0:
        final = conf_add/len(obj_pixels)
    else: 
        return -999,-999

    return final,obj_pixels


if __name__ == "__main__":
    pass
    