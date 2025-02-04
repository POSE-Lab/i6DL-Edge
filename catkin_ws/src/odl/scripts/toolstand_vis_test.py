from ToolstandDetector_stable import *
import cv2

script_dir = os.path.dirname(__file__)
imagepath = "/root/sandbox/test_images/image_raw2.png"
savepath = "/root/sandbox/test_images/"
tD = ToolstandDetector(os.path.join(script_dir,"config_toolstand.yaml"))
tD.K = [377.6746826171875, 0.0, 314.96502685546875, 0.0, 377.2413330078125, 252.47396850585938, 0.0,
0.0, 1.0]
rgb = cv2.imread(imagepath)

tD.detect(rgb, WS_id = "-22")
tD.visualize(savepath, imgname = "result4")
