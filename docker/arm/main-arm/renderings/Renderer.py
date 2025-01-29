from ast import Try
from dataclasses import dataclass
import imp
from pickletools import uint8
from tkinter import Frame
from cv2 import Mat, sortIdx
import numpy as np
from OpenGL.GL import *
import cv2
import pyrr
import os
from PIL import Image
import png


@dataclass
class Framebuffer:
    ID: int = None
    colorBuffer: int = None
    width: int = None
    height: int = None


@dataclass
class params:
    distances: list[float]  # meters - has to be converted to milimeters
    startPhi: int
    stopPhi: int
    startTheta: int
    stopTheta: int

    phiStep: int
    ThetaStep: int

    mode: list[str]


@dataclass
class BufferHolder:
    # texture
    VAO: uint8
    indices: list[np.uint32]
    shader_program: uint8
    texture: uint8 = None

    # depth - depth same indices
    VAO_depth: uint8 = None
    shader_depth_program: uint8 = None


draw_color = [1.0, 1.0, 1.0, 1.0]


def CalcRenderedImagesNumber(len_dist: int, len_phi_angles: int, len_theta_angles: int):
    """
    Calculates the total number of imagse to be rendered based on the render parameters.
    Note that the output number is refering to the number of scenes rendered. (You may render
    both texture and depth. In this case the total images are imNum*2)
    """
    imNum = len_dist * len_phi_angles * (len_theta_angles+1)
    return imNum


def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def CreateFramebuffer(FBO_WIDTH, FBO_HEIGHT, internalFormat, format, type):
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
    print("Framebuffer : "+str(framebuffer))
    # color attachemnt - texture
    # empty texture that will be filled by rendering to the buffer
    textureColorBuffer = glGenTextures(1)
    print("Colorbuffer: "+str(textureColorBuffer))
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer)
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, FBO_WIDTH,
                 FBO_HEIGHT, 0, format, type, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, textureColorBuffer, 0)

    # create a renderbuffer object for depth and stencil attachment
    RBO = glGenRenderbuffers(1)
    print("RenderBuffer: "+str(RBO))
    glBindRenderbuffer(GL_RENDERBUFFER, RBO)
    glRenderbufferStorage(
        GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, FBO_WIDTH, FBO_HEIGHT)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO)

    # check if the framebuffer was properly created and complete
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("ERROR::FRAMEBUFFER: Framebuffer is not complete.\n")
        print(hex(glCheckFramebufferStatus(GL_FRAMEBUFFER)))
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    FRB = Framebuffer(framebuffer, textureColorBuffer, FBO_WIDTH, FBO_HEIGHT)

    return FRB


def CaptureFramebufferScene(framebuffer: Framebuffer, savePath):
    """
    Captures the scene and saves it in savePath specified.
    Parameters:
        - framebuffer : of type Framebuffer object. Defines the framebuffer
        the function will sample the scene
        - savePath: the output path to save the capture (Also defines the format i.e. png/jpg)
    """
    # bind the framebuffer that will be sampled and its colorbuffer to read mode
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer.ID)
    # glReadBuffer(framebuffer.colorBuffer)

    pixels = glReadPixels(0, 0, framebuffer.width,
                          framebuffer.height, GL_RGBA, GL_UNSIGNED_BYTE)

    # invert the image, pixel -> 1D array
    # TODO:

    # Different hundling when user requests capturing of a depth image

    try:
        img = np.frombuffer(pixels, dtype=np.uint8)

        img = np.reshape(img, (framebuffer.height, framebuffer.width, 4))
        rev_pixels = img[::-1, :]
        img = cv2.cvtColor(rev_pixels, cv2.COLOR_BGRA2RGB)
        print(img.size)
        cv2.imwrite(savePath, img)
    except cv2.error as e:
        print(e)

    # bind to the deafault framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return img


def ProjectFramebuffer(sourceFRB, windowSize):
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


def RenderRGBD(framebuffer: Framebuffer, camera, params: params, savePath, buffHolder: BufferHolder):
    """
    Renders to an ofscreen framebuffer using the parameters that are defined under
    "Render images" menu.
    Parameters:
        - Framebuffer to rendere to(Framebuffer object type)
        - Camera object. See Camera.py. Used to updated the view
        - params : the parameters used to perform the rendering (path,phi,theta,validation set)

    Note: If Depth mode is enabled different framebuffer hundling may be required
    TODO: Depth, params maybe build a struct for eazy access?
    """

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.ID)

    progress = 0

    for dist in params.distances:
        for theta in range(params.startTheta, params.stopTheta + params.ThetaStep, params.ThetaStep):
            for phi in range(params.startPhi, params.stopPhi, params.phiStep):

                camera.setParams(phi, theta, dist)
                camera.update_camera_parameters()
                print("Loop:: Theta: "+str(theta)+" Phi: " +
                      str(phi)+" Distance : "+str(dist))

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # NO TEXTURE
                if "Texture" in params.mode:
                    glBindTexture(GL_TEXTURE_2D, buffHolder.texture)
                    glUseProgram(buffHolder.shader_program)
                    transformLoc = glGetUniformLocation(
                        buffHolder.shader_program, "transform")
                    color = glGetUniformLocation(
                        buffHolder.shader_program, "color")

                    projection = pyrr.Matrix44.perspective_projection(
                        45, float(1920)/1080., 100, 1000000)

                    glUniformMatrix4fv(transformLoc, 1, GL_FALSE,
                                       projection*camera.tranformation)
                    glUniform4fv(color, 1, pyrr.Vector4(draw_color))

                    glBindVertexArray(buffHolder.VAO)
                    glDrawElements(GL_TRIANGLES, len(
                        buffHolder.indices), GL_UNSIGNED_INT, None)

                    pixels = glReadPixels(0, 0, framebuffer.width,
                                          framebuffer.height, GL_RGB, GL_UNSIGNED_BYTE)

                    try:
                        img = np.frombuffer(pixels, dtype=np.uint8)

                        img = np.reshape(
                            img, (framebuffer.height, framebuffer.width, 3))
                        rev_pixels = img[::-1, :]
                        img = cv2.cvtColor(rev_pixels, cv2.COLOR_BGR2RGB)
                        print(img.size)
                        if not os.path.exists(savePath + "/rgb"):
                            os.makedirs(savePath + "/rgb")
                        cv2.imwrite(savePath + "/rgb" + "/" +
                                    str(progress)+".png", img)
                    except cv2.error as e:
                        print(e)

                if "Depth" in params.mode:

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glEnable(GL_DEPTH_TEST)
                    glDisable(GL_CULL_FACE)

                    glUseProgram(buffHolder.shader_depth_program)

                    projection = pyrr.Matrix44.perspective_projection(
                        45, float(1920)/1080., 100, 1000000)

                    # define uniforms and projection matrix
                    transformLoc = glGetUniformLocation(
                        buffHolder.shader_depth_program, "transform")
                    projectionLoc = glGetUniformLocation(
                        buffHolder.shader_depth_program, "projection")
                    glUniformMatrix4fv(
                        transformLoc, 1, GL_FALSE, camera.tranformation)
                    glUniformMatrix4fv(
                        projectionLoc, 1, GL_FALSE, projection)
                    # bind depth VAO
                    glBindVertexArray(buffHolder.VAO_depth)
                    glDrawElements(GL_TRIANGLES, len(
                        buffHolder.indices), GL_UNSIGNED_INT, None)

                    img_from_buffer = np.zeros(
                        (framebuffer.height, framebuffer.width, 3), dtype=np.float32)

                    glReadPixels(0, 0, framebuffer.width,
                                 framebuffer.height, GL_RGB, GL_FLOAT, img_from_buffer)

                    try:
                        img_from_buffer.shape = (
                            framebuffer.height, framebuffer.width, 3)
                        img_from_buffer = img_from_buffer[::-1, :]
                        img_from_buffer = img_from_buffer[:, :, 0]

                        if not os.path.exists(savePath + "/depth"):
                            os.makedirs(savePath + "/depth")

                        save_depth(savePath + "/depth" + "/" +
                                   str(progress)+".png", img_from_buffer)
                        # cv2.imwrite(savePath + "/depth" + "/" +
                        #            str(progress)+".png", im_uint16)
                    except cv2.error as e:
                        print(e)

                progress += 1
                # hundle depth rendering

    # glBindFramebuffer(GL_FRAMEBUFFER, 0)
