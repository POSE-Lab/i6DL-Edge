"""
This class contains the following methods:
    - ReadShaderFromFile(fielanamem,type)
"""
import os
import glfw
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGL.arrays import *
from OpenGL.GL import shaders


class Shader():
    
    def __init__(self,vertexfilename,fragmentfilename,geometry_filename=None) -> None:
        self.vertexfilename = vertexfilename
        self.fragmentfilename = fragmentfilename
        self.geometryFilename = geometry_filename
        
    def printID(self):
        print(self.filename)

    def readShadersFromFile(self):
        
        with open(self.vertexfilename,'r') as file:
            shader = file.read()
            self.vertexstring = shader
        with open(self.fragmentfilename,'r') as file:
            shader = file.read()
            self.fragmentstring = shader
        if self.geometryFilename is not None:
            with open(self.geometryFilename,'r') as file:
                shader = file.read()
                self.geometrystring = shader
        
    def compileShader(self):
        vertex = shaders.compileShader(self.vertexstring,GL_VERTEX_SHADER)
        fragment = shaders.compileShader(self.fragmentstring,GL_FRAGMENT_SHADER)
        if self.geometryFilename is not None:
            geometry = shaders.compileShader(self.geometrystring,GL_GEOMETRY_SHADER)
            return vertex,fragment,geometry
        else:
            return vertex,fragment,None

    def compileProgram(self,vertex_shader,fragment_shader,geometry_shader):
        if self.geometryFilename is not None:
            program = shaders.compileProgram(vertex_shader,fragment_shader,geometry_shader)
        else:
            program = shaders.compileProgram(vertex_shader,fragment_shader)
        
        return program

    def UseProgram(self,programmID):
        shaders.glUseProgram(programmID)

    


    
    
