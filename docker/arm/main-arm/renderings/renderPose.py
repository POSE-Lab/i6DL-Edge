from OpenGL.GL import *
import numpy as np
import pyrr
import glfw
from pyrr import Matrix44, Vector3
import time
from Shader import Shader
from Renderer import *
from plyfile import PlyData
from time import time 
from PIL import Image,ImageFilter,ImageEnhance
from absl import logging
from absl import flags
from absl import app

SCRN_WIDTH, SCRN_HEIGHT = 640,480
FBO_WIDTH, FBO_HEIGHT = SCRN_WIDTH, SCRN_HEIGHT

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'rt','1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16',
    'RT pose matrix'
)
flags.DEFINE_string(
    'rt_file','',
    'Provide RT through file. Used with multiImage'
)
flags.DEFINE_bool(
    'multiImage',False,
    "Batch of images or single image"
)
flags.DEFINE_string(
    'img_Path','',
    "Image path if multiImage=True or image directory if multiImage=False"
)
flags.DEFINE_float(
    'op',0.7,
    "Opacity of the overlayed pose"
)
flags.DEFINE_string(
    'Kapa','',
    'Choose K from available Ks'
)


def binarize(img,thresh):


  #convert image to greyscale
  img=img.convert('L') 

  width,height=img.size

  #traverse through pixels 
  for x in range(width):
    for y in range(height):

      #if intensity less than threshold, assign white
      if img.getpixel((x,y)) < thresh:
        img.putpixel((x,y),0)

      #if intensity greater than threshold, assign black 
      else:
        img.putpixel((x,y),255)

  return img

def loadPLY(modelPath):
    modeldata = PlyData.read(modelPath)
    
    vertices = np.array(modeldata['vertex'][:])
    vertices = vertices.view((vertices.dtype[0], len(vertices.dtype.names)))
    face_ind = map(lambda o: np.array(o) ,np.array(modeldata['face'].data['vertex_indices'])) 
    
    return vertices.astype(np.float32).flatten(),np.array(list(face_ind)).flatten()

def main(argv):
    glfw.init()
    window = glfw.create_window(SCRN_WIDTH,SCRN_HEIGHT,'RenderPose.py',None,None)
    glfw.hide_window(window)
    glfw.make_context_current(window)
   
    time_s = time()
    cube,cube_indices = loadPLY("../obj_000002_normal.ply")  #H:/FELICE/6dPAT/screwdriver_2_texture/att3/model_filled_last_noUV.ply
    time_stop = time() - time_s
    print("Loading took : "+str(time_stop))
    print(cube.itemsize)

    #shaders = Shader("vertex_shader.txt", "fragment_shader.txt", None)
    shaders = Shader("basic_lighting_vrt.txt", "basic_lighting.txt", None)
    shaders.readShadersFromFile()
    vertex, fragment, geometry = shaders.compileShader()

    shader = shaders.compileProgram(vertex, fragment, geometry)
    
    VAO_lines = glGenVertexArrays(1)
    VBO_lines = glGenBuffers(1)
    VAO_triangles = glGenVertexArrays(1)
    VBO_triangles = glGenBuffers(1)
    EBO_triangles = glGenBuffers(1)
    
    glBindVertexArray(VAO_triangles)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_triangles)
    glBufferData(GL_ARRAY_BUFFER, cube.nbytes, cube, GL_STATIC_DRAW)
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_triangles)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)
    
    
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 6, ctypes.c_void_p(0))
    
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, cube.itemsize * 6, ctypes.c_void_p(12))
    
    
    rgb_framebuffer = CreateFramebuffer(
        FBO_WIDTH, FBO_HEIGHT, GL_RGB32F, GL_RGBA, GL_FLOAT)

    
    glEnable(GL_DEPTH_TEST)
    
    prevTime = glfw.get_time()
    frameCount = 0
    totFrames = 0
    fps_list = [0.0]
    FPS=0
    
    #while not glfw.window_should_close(window):
    start_render = time()
    total_time = time()
    current_time = glfw.get_time()
    frameCount += 1

    if (current_time - prevTime >= 1.0):
        FPS = frameCount
        totFrames += 1
        fps_list.append(FPS)

        if totFrames > 20:
            fps_list.pop(0)

        frameCount = 0
        prevTime = current_time
    
    print("FPS: ",FPS)
    glBindFramebuffer(GL_FRAMEBUFFER, rgb_framebuffer.ID)
    
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)
    
    
    objectColor = glGetUniformLocation(shader, "objectColor")
    lightColor = glGetUniformLocation(shader, "lightColor")
    lightPos = glGetUniformLocation(shader, "lightPos")
    viewPos = glGetUniformLocation(shader, "viewPos")
    
    glUniform3fv(objectColor,1,pyrr.Vector3([1.0, 0.5, 0.31]))
    glUniform3fv(lightColor,1, pyrr.Vector3([1.0, 1.0, 1.0]))
    
    
    
    
    model = glGetUniformLocation(shader, "model")
    view = glGetUniformLocation(shader, "view")
    projection = glGetUniformLocation(shader, "projection")
    
    #color = glGetUniformLocation(shader, "color")
    view_ = pyrr.Matrix44.identity()
    model_ = pyrr.Matrix44.identity()
    trf = [[1.0,0.0,0.0,0.2],
            [0.0,1.0,0.0,0.0],
            [0.0,0.0,1.0,0.0],
            [0.0,0.0,0.0,1.0]]
    trf = np.array(trf).astype(np.float32)
    projection_ = pyrr.Matrix44.perspective_projection(
        45, float(1280)/720., 0.01, 1000.0)
    
    #print(la)
    
    fx = 634.3643798828125
    fy = 633.6349487304688
    cx = 637.8005981445312
    cy = 364.95831298828125

    K = {
    'K1': [634.3643798828125, 0.0, 637.8005981445312, 0.0, 633.6349487304688, 364.95831298828125, 0.0, 0.0, 1.0],
    'K2': [642.6622314453125, 0.0, 631.1490478515625, 0.0, 641.2005615234375, 363.1187744140625, 0.0, 0.0, 1.0],
    'K3': [636.4381103515625, 0.0, 637.8005981445312, 0.0, 635.706298828125, 364.95831298828125, 0.0, 0.0, 1.0],
    'K_tests': [642.836426, 0.000000, 631.149048, 0.000000, 641.374451, 363.118774, 0.000000, 0.000000, 1.000000]
}
    print("kkkk ",np.array(K['K_tests']))
    K = np.array(K['K_tests']).reshape(3,3).astype(float)
    
    
    nearP = 1
    farP = 5000
    
    
    proj = [ 2 * K[0,0] / float(FBO_WIDTH),     -2 * 0 / float(FBO_WIDTH) ,     0,    0,
            0,                       2 * K[1,1] / float(FBO_HEIGHT),      0,   0,
            1 - 2 * (K[0,2] / float(FBO_WIDTH)),      2 * (K[1,2] / float(FBO_HEIGHT)) - 1,   -(farP + nearP) / (farP - nearP),       -1,
            0,                       0,                      2 * farP * nearP / (nearP - farP),       0
            ]
    proj = np.array(proj).reshape(4,4)

    RT = np.fromstring(FLAGS.rt,sep=',').reshape(4,4).astype(float)
    INV_MATR  = np.array([1.0 , 0.0, 0.0, 0.0,
        0.0,-1.0,0.0,0.0,
        0.0,0.0,-1.0,0.0,
        0.0, 0.0, 0.0, 1.0]).reshape((4,4))
   
    
    print("RT_BEFORE THE TRNSPOSE, ",RT)
    
    glUniform3fv(viewPos,1,pyrr.Vector3([-RT[0,-1], -RT[1,-1], -RT[2,-1]]))
    
    RT = np.transpose(RT)
    RT = np.dot(RT,INV_MATR)
    res = np.dot(np.array([RT[0,-1],RT[1,-1],RT[2,-1],1.0]),np.linalg.inv(RT))
    
    glUniform3fv(lightPos,1,pyrr.Vector3([res[0], res[1] , res[2]]))
    print("****res*** ",res)
    la = pyrr.Matrix44.from_translation(
        translation=(RT[0,-1], -RT[1,-1], -RT[2,-1]))
    print("RT_AFTER THE TRNSPOSE, ",RT)
    #rot = np.array(rot).reshape(3,3)
    
    model_ = pyrr.Matrix44.identity()
    model_ = model_
    glUniformMatrix4fv(model, 1, GL_FALSE,model_)
    glUniformMatrix4fv(view, 1, GL_FALSE,
                        RT)
    glUniformMatrix4fv(projection, 1, GL_FALSE,
                        proj)
    
    glBindVertexArray(VAO_triangles)
    glDrawElements(GL_TRIANGLES, len(cube_indices), GL_UNSIGNED_INT, None)
    
    ProjectFramebuffer(rgb_framebuffer, (SCRN_WIDTH, SCRN_HEIGHT))
    mask = CaptureFramebufferScene(rgb_framebuffer, "lala.png")
    glfw.swap_buffers(window)
    
    stop_render = time() - start_render
    print("Rendering time ",str(stop_render))
    #break
         
    
    time_mask = time()
    foreground = mask.astype(float)
    print(FLAGS.img_Path)
    background = cv2.imread('/home/panos/Downloads/results_2023_05_17/Test_2023_05_17_161646/Obj_2/image_raw.png').astype(float)
    
    
    # create a mask for the object
    mask = cv2.threshold(foreground,1,np.ceil(FLAGS.op*255),cv2.THRESH_BINARY)[1]
   
    alpha = mask/255.
    
    foreground = cv2.multiply(alpha,foreground)
    
    background = cv2.multiply(1.0-alpha,background)
    
    outImage = cv2.add(foreground, background); 

    print("Time mask: ",str(time() - time_mask))
    time_cv = time()

    result = outImage.copy()
    
    maskc = cv2.cvtColor(mask.astype(np.uint8),cv2.COLOR_RGB2GRAY)
    contours = cv2.findContours(maskc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    
    x,y,w,h = cv2.boundingRect(contours[0])
    cv2.rectangle(result, (x, y), (x+w, y+h), ((0, 165, 255)), 1)
    cv2.rectangle(result, (x, y-2), (x+200, y-25), (0, 165, 255), -1)
    result = cv2.putText(result,"Car-door-panel: 99.0",(x,y-2),fontScale=0.7,color=(255,255,255),thickness=2,fontFace=cv2.LINE_AA)
    print("x,y,w,h:",x,y,w,h)
    
    print("Time cv: ", str(time() - time_cv))
    
   
    time_save_cv = time()
    
    cv2.imwrite('/home/panos/Downloads/results_2023_05_17/Test_2023_05_17_161646/Obj_2/result.png',result)  
   
    print("Time save cv: ",str(time() - time_save_cv))

    
    print("Total time: ",str(time() - start_render))

if __name__ =="__main__":
    app.run(main)