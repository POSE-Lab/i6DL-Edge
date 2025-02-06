from OpenGL.GL import *
import numpy as np
import pyrr
import glfw
from pyrr import Matrix44,Vector3
from dataclasses import dataclass
from pickletools import uint8
import cv2 as cv

# custom utility functions for rendering
from Shader import Shader

# load object model
from plyfile import PlyData

from time import time
from PIL import Image,ImageFilter,ImageEnhance

from absl import logging
from absl import flags
from absl import app

from typing import List

@dataclass
class Framebuffer:
    ID: int = None
    colorBuffer: int = None
    width: int = None
    height: int = None


@dataclass
class params:
    distances: List[float]  # meters - has to be converted to milimeters
    startPhi: int
    stopPhi: int
    startTheta: int
    stopTheta: int

    phiStep: int
    ThetaStep: int

    mode: List[str]


@dataclass
class BufferHolder:
    # texture
    VAO: uint8
    indices: List[np.uint32]
    shader_program: uint8
    texture: uint8 = None

    # depth - depth same indices
    VAO_depth: uint8 = None
    shader_depth_program: uint8 = None

def load_model(model_path):
    """Loads a 3D model provided with a ply format.

    Args:
        model_path (str): Path to 3D model

    Returns:
        tuple: (Model verices, face indices) 
    """
    modeldata = PlyData.read(model_path)
    
    vertices = np.array(modeldata['vertex'][:])
    vertices = vertices.view((vertices.dtype[0], len(vertices.dtype.names)))
    face_ind = map(lambda o: np.array(o) ,np.array(modeldata['face'].data['vertex_indices'])) 
    
    return vertices.astype(np.float32).flatten(),np.array(list(face_ind)).flatten()
    
class Renderer:
    def __init__(self,bufferSize=(1280,720)) -> None:
        
        self.FBO_size = bufferSize
        self.initGLFW(self.FBO_size)
        
    def initGLFW(self,FBO_size):
        
        glfw.init()
        self.window = glfw.create_window(FBO_size[0],FBO_size[1],'RenderPose.py',None,None)
        glfw.hide_window(self.window)
        glfw.make_context_current(self.window)
        
    def load_shaders(self,vrt,frg,geo):
        """Load and compiles shaders given the filename of each one to disc.
        Geometry shader can be None. Vertex and Fragment are mandatory. It must be called
        as many times as the required rendering modes i.e. rendering wtih dexture, depth, wireframe etc.

        Args:
            vrt (str): Vertex shader path
            geo (str): Geometry shader path
            frg (str): Fragment shader path
        """
        shaders = Shader(vrt,frg,geo)
        shaders.readShadersFromFile()
        vertex,fragment,geometry = shaders.compileShader()
        
        self.shader_programm = shaders.compileProgram(vertex,fragment,geometry)
        
    def create_buffers(self,modelData,faceIndices,attrs):
        
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)
        
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER,self.VBO)
        glBufferData(GL_ARRAY_BUFFER,modelData.nbytes,modelData,GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 faceIndices.nbytes, faceIndices, GL_STATIC_DRAW)
    
        stride = attrs[0]*attrs[1]
        for i in range(attrs[0]):
            offset = attrs[1]*i*4
            glEnableVertexAttribArray(i)
            glVertexAttribPointer(i,attrs[1],GL_FLOAT,GL_FALSE,modelData.itemsize * stride,ctypes.c_void_p(0 if i==0 else offset))
    
    def CreateFramebuffer(self, internalFormat, format, type):
        """
            Creates and returs(framebuffer id) a framebuffer.
            Parameters:
            - FBO_WIDTH : Width of the framebuffer to create (pixels)
            - FBO_HEIGHT: Height of the framebuffer (pixels)
            - internalFormat :
            - format :
            - type :
            Last three parameters are associated with glTexImage2D()
        """
        framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
        print("[vis_pose_wraper] Framebuffer : ",str(framebuffer))
        # color attachemnt - texture
        # empty texture that will be filled by rendering to the buffer
        textureColorBuffer = glGenTextures(1)
        print("[vis_pose_wraper] Colorbuffer: "+str(textureColorBuffer))
        glBindTexture(GL_TEXTURE_2D, textureColorBuffer)
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, self.FBO_size[0],
                    self.FBO_size[1], 0, format, type, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, textureColorBuffer, 0)

        # create a renderbuffer object for depth and stencil attachment
        RBO = glGenRenderbuffers(1)
        print("[vis_pose_wraper] RenderBuffer: "+str(RBO))
        glBindRenderbuffer(GL_RENDERBUFFER, RBO)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, self.FBO_size[0], self.FBO_size[1])
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO)

        # check if the framebuffer was properly created and complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("ERROR::FRAMEBUFFER: Framebuffer is not complete.\n")
            print(hex(glCheckFramebufferStatus(GL_FRAMEBUFFER)))
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.framebuffer = Framebuffer(framebuffer, textureColorBuffer, self.FBO_size[0], self.FBO_size[1])


    def CaptureFramebufferScene(self, savePath=None,saveRendered=False):
        """
        Captures the scene and saves it in savePath specified.
        Parameters:
            - framebuffer : of type Framebuffer object. Defines the framebuffer
            the function will sample the scene
            - savePath: the output path to save the capture (Also defines the format i.e. png/jpg)
        """
        # bind the framebuffer that will be sampled and its colorbuffer to read mode
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.framebuffer.ID)
        # glReadBuffer(framebuffer.colorBuffer)

        pixels = glReadPixels(0, 0, self.framebuffer.width,
                            self.framebuffer.height, GL_RGBA, GL_UNSIGNED_BYTE)

        # invert the image, pixel -> 1D array
        # TODO:

        # Different hundling when user requests capturing of a depth image

        try:
            img = np.frombuffer(pixels, dtype=np.uint8)

            img = np.reshape(img, (self.framebuffer.height, self.framebuffer.width, 4))
            rev_pixels = img[::-1, :]
            img = cv.cvtColor(rev_pixels, cv.COLOR_BGRA2RGB)
            if saveRendered:
                cv.imwrite(savePath, img)
        except cv.error as e:
            print(e)

        # bind to the deafault framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return img.astype(float)


    def ProjectFramebuffer(self,sourceFRB, windowSize):
        """
        Blits a framebuffer to screen or another framebuffer. Note that if the sizes not match
        linear interpolation is performed(that may lead to poor image quality)
        Parameters:
            - sourceFRB : Source framebuffer (framebuffer to perform the sampling)
            - destFRB : Destination framebuffer to project the above to.
            - size : Tuple declarting the (widht,height) of the output framebuffer to project
        """
        glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFRB.ID)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

        glBlitFramebuffer(0, 0, sourceFRB.width, sourceFRB.height, 0, 0,
                        windowSize[0], windowSize[1], GL_COLOR_BUFFER_BIT, GL_LINEAR)
        
    def setUniformVariables(self,shader_programm,variable_dict):
        """Set uniform variables in shaders by providing a variable dict with the following 
        format:
        
        {uniform_name: "name",uniform_value: value,type: type}

        Args:
            variable_dict (dict): Dictionary containing the variable name in shaders and their value
        """
        locs ={}
        for variable in variable_dict:
           
            locs[variable] = glGetUniformLocation(shader_programm,variable)
            if variable_dict[variable]["type"] == 'glUniformMatrix4fv':
                eval(variable_dict[variable]["type"])(locs[variable],1,GL_FALSE,variable_dict[variable]["value"])
            else:
                eval(variable_dict[variable]["type"])(locs[variable],1,variable_dict[variable]["value"])

    @staticmethod
    def calc_chars_width(text):
        return len(text) * 17
    def makeOverlay(self,capturedImage,renderedImage,objectName,confidense,pose_status,savePath,opacity=0.7):
        """CapturedImage represents the background while the renderedImage the 
        rendered object in the estimated pose.

        Args:
            capturedImage (np.array image): Captured image from camera
            renderedImage (np.array image): Rendered object pose
        """
        # overlay the rendered pose on top of the captured image
        mask = cv.threshold(renderedImage,1,np.ceil(opacity*255),cv.THRESH_BINARY)[1]
        alpha = mask/255.
        forground = cv.multiply(alpha,renderedImage)
        background = cv.multiply(1.0-alpha,capturedImage)
        outImage = cv.add(forground,background)
        
        # use opencv to draw the bounding box around the object and set the label
        result = outImage.copy()
        maskc = cv.cvtColor(mask.astype(np.uint8),cv.COLOR_RGB2GRAY)
        contours = cv.findContours(maskc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        
        x,y,w,h = cv.boundingRect(contours[0])

        text = objectName+" : "+str(confidense)
        logging.debug(f"[vis_pose_wraper] Pose Status: {pose_status}")
        if pose_status.strip('\n') == "accepted":
            cv.rectangle(result, (x, y), (x+w, y+h), ((0, 255, 0)), 2)
            cv.rectangle(result, (x, y-2), (x+self.calc_chars_width(text), y-25), (0, 255, 0), -1)
        else:
            cv.rectangle(result, (x, y), (x+w, y+h), ((0, 0, 255)), 2)
            cv.rectangle(result, (x, y-2), (x+self.calc_chars_width(text), y-25), (0, 0, 255), -1)
        
        result = cv.putText(result,text,(x,y-2),fontScale=1.0,
                            color=(255,255,255),thickness=2,fontFace=cv.LINE_AA)

        cv.imwrite(savePath,result)

def watchDir():
    pass

def getRenderingParams():
    pass
