from matplotlib import image
from matplotlib import pyplot as plt
from numpy import loadtxt
import numpy as np
from matplotlib.patches import Rectangle
import argparse
import os
import glob
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--corPath", type=str, required=True)
parser.add_argument("--imgPath",type=str,required=True)
parser.add_argument("--resPath",type=str,required=True)
args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)

def draw_2DBB(seg_pixels):
    # find most left and up pixel

    pass


def draw_point(x, y, marker, color):
    plt.plot(x, y, marker=marker, color=color, markersize=2)


def load_image(img_path):
    return image.imread(img_path)


def read_inliers_indices(file):
    temp2 =[]
    with open(file, 'r') as f:
        temp = f.readlines()[8:-1]
        #print(temp)print
        for t in temp:
            #print(t)
            temp2.append(t[0:-2].strip("\n").split(" "))
        #print(temp2)
    return np.array(temp2).astype(float)


def save_result(img,path,inliers,total):
    plt.title("Total corresp: {}, Inliers: {}, Ratio(I/O): {:.2f}".format(total,inliers,inliers/total))
    plt.imshow(img)
    plt.savefig(path)
    


if __name__ == "__main__":

    for d in glob.glob(args.corPath+"/*"):
        #print(d)
        x_AR = []
        y_AR = []
        fig, ax = plt.subplots()
        l = read_inliers_indices(os.path.join(args.corPath,d))
        #print(l)
        for item in l:
                #print(item)
                x, y = item[1],item[2]
                #print(x, y)
                x_AR.append(x)
                y_AR.append(y)
        DL = (min(x_AR), min(y_AR))
        UR = (max(x_AR), max(y_AR))

        total_corresp = len(l[:])
        inliers = np.where(l[:][:,0]==1.0)
        vals = l[inliers[0]]
        #print(len(vals))
        #print(vals)
        print("Mean inlier 2D Object confidense: ",np.mean(vals[:,-2]))
        print("Mean inlier fragment confidense: ",np.mean(vals[:,-1]))
        print("Mean inlier total confidense: ",np.mean(vals[:,-3]))
        print(np.mean(vals[:,-1]))
        print("INLIERS ",len(inliers[0]))
        ax.set_xlim(float(DL[0])-20, float(UR[0])+20)
        ax.set_ylim(float(DL[1])-20, float(UR[1])+20)
        plt.gca().invert_yaxis()

        ax.add_patch(Rectangle((float(DL[0]), float(DL[1])),
                               float(UR[0])-float(DL[0]),
                               float(UR[1])-float(DL[1]),
                               edgecolor='red',
                               fill=False))
        """
        print(args.imgPath +'/0000'+(os.path.basename(d).split('_')[-1]).strip('.txt') + '/rgb/'+os.path.basename(d).split('_')[0]+'.png')
        img = load_image(args.imgPath +'/0000'+(os.path.basename(d).split('_')[-1]).strip('.txt') + '/rgb/'+os.path.basename(d).split('_')[0]+'.png')
        """
        img = load_image(args.imgPath)
        # draw outliers
        draw_inliers=0
        counter = 0
        for i, j in zip(x_AR, y_AR):
            if int(l[counter][0]) == 1:
                draw_point(float(i), float(j), marker='x', color='aquamarine')
                draw_inliers += 1
            else:
                draw_point(float(i), float(j), marker='x', color='red')
            counter += 1
        print(draw_inliers)
        """
        save_result(img,args.resPath+"/inliers_"+os.path.basename(d).split('_')[0]+".png",len(inliers[0]),total_corresp)
        """
        save_result(img,args.resPath,len(inliers[0]),total_corresp)
