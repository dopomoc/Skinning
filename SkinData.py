'''
SkinData Class - v1.0

Utility class for Skinning BVH Files
Written by Darren Cosker 2020

Requires
- OBJ object (and OBJData module)
- BVH object (and BVHData module)
- Skinning weights - IMPORTANT! Assumes loaded as a .mat array (Matlab)
    Easy to modify - just make self.skinWeights a verts X joints array

Given BVH and OBJ can perform Rigid Skinning or Linear Blend Skinning (LBS)

'''


import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import BVHData as bvh
import OBJData as obj
from scipy.io import loadmat
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class SkinData:
    
    def __init__(self):
        self.skinWeights = []
        self.jointTFormList = []
        self.allVertsX = []
        self.allVertsY = []
        self.allVertsZ = []
        self.npFacesSplit = []
        self.minX = 10000
        self.maxX = -10000
        self.minY = 10000
        self.maxY = -10000
        self.minZ = 10000
        self.maxZ = -10000
        self.animationPreview = []
                

    def loadSkin(self, skinningFileName):
        importedSkinData = loadmat(skinningFileName)  # NB modify for other formats      
        self.skinWeights = importedSkinData['data']   # NB modify for other formats      
        
        
    def animateSkin(self, bvhObject, objObject, frameStep=1, skinning='RS', draw = 'Poly'):
        
        '''
        # Requires
        # - bvhObject - bvh object
        # - objObject - obj object
        # - frameStep  - drawing interval
        # - skinning - skinning method - Rigid Skinning (RS) or Linear Blend Skinning (LBS)
        # - draw - drawing method: dots (Dots) or polygonal (Poly). NB dots is currently much faster
        '''
        
        skinWeights = np.array(skinObject.skinWeights)
            
        # Get joint transforms for this frame in a list
        frame = 0
        frameStart = 0
        frameEnd =  bvhObject.totalFrames    
            
        # Draw the output frame
        npFaces = np.array(objObject.faces)        
        [self.npFacesSplit.append([int(npFaces[i][0].split('/')[0])-1,int(npFaces[i][1].split('/')[0])-1,int(npFaces[i][2].split('/')[0])-1]) for i in range(objObject.numFaces)]

        if skinning == 'RS':
            # Rigid Skinning (linear blend skinning to follow, Pose Space Deformation after that maybe!)
            # Algorithm:
            # for each vertex in rest pose mesh
            #   What is (major, for rigid skinning in case >1 bone weight give) joint?
            #   Get bind pose (inverse of local>world transform for that bone)
            #   Apply bind pose to the vertex
            #   Apply new local>world transform to the vertex
            print('Rigid Skinning..')                      
            
            for frame in range(frameStart,frameEnd,frameStep):
                           
                self.jointTFormList = []
                self.getJointTFormList(bvhObject, frame) 
                
                npVerticesX = np.zeros(objObject.numVerts)
                npVerticesY = np.zeros(objObject.numVerts)
                npVerticesZ = np.zeros(objObject.numVerts)
                        
                # Take current pose and apply to rest pose mesh
                for vertIter in range(objObject.numVerts):
                                
                    thisVert = objObject.vertices[vertIter]
                    thisVert = [float(thisVert[i]) for i in range(len(thisVert))]
                    thisVert.append(float(1))
                            
                    thisBoneIdx = np.argmax(skinWeights[vertIter,:])   
                                        
                    tForm = np.matmul(self.jointTFormList[thisBoneIdx],bvhObject.bindTfrms[thisBoneIdx])
                    thisVert = np.matmul(tForm,thisVert)

                    npVerticesX[vertIter] = float(thisVert[0])
                    npVerticesY[vertIter] = float(thisVert[1])
                    npVerticesZ[vertIter] = float(thisVert[2])            
                
                self.allVertsX.append(npVerticesX)
                self.allVertsY.append(npVerticesY)
                self.allVertsZ.append(npVerticesZ)
                    
                
            # Get min and max values for x, y and z axis    
            self.getAnimationBounds()  
            
            if draw == 'Poly':
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.set_xbound(self.minX,self.maxX)
                ax.set_ybound(self.minY,self.maxY)
                ax.set_zbound(self.minZ,self.maxZ)

                self.animationPreview = []

                for frame in range(frameStart,frameEnd,frameStep):

                    verts = []
                    for face in range(len(self.npFacesSplit)):
                        tF = self.npFacesSplit[face]                        
                        
                        vX = [self.allVertsX[frame][tF[0]], self.allVertsX[frame][tF[1]], self.allVertsX[frame][tF[2]]]
                        vY = [self.allVertsY[frame][tF[0]], self.allVertsY[frame][tF[1]], self.allVertsY[frame][tF[2]]]
                        vZ = [self.allVertsZ[frame][tF[0]], self.allVertsZ[frame][tF[1]], self.allVertsZ[frame][tF[2]]]
                        
                        verts.append(list(zip(vX,vZ,vY))) # flip Y and Z to match bvh axis
    
                    self.animationPreview.append(verts)
                
                self.polyObj = Poly3DCollection(self.animationPreview[0], linewidths=0.1)
                self.polyObj.set_edgecolor('cyan')

                ax.add_collection3d(self.polyObj)
                self.ani = animation.FuncAnimation(fig, self.drawPoly)
                plt.show()

                # Create gif movie
                outputFileName = "{}-{}-{}.gif"
                outputFileName.format(bvhObject.fileName,draw,skinning)
                print('Writing ',outputFileName.format(bvhObject.fileName,draw,skinning))
                self.ani.save(outputFileName.format(bvhObject.fileName,draw,skinning), writer=animation.PillowWriter(fps=30))
                


            if draw == 'Dots':                    
                # Draw animation
                self.fig = plt.figure()
                #mng = plt.get_current_fig_manager()
                #mng.window.showMaximized()
                # NB swapping Y and Z as Y is up and not Z
                self.ax = self.fig.add_subplot(projection="3d",xlim=(self.minX,self.maxX), ylim=(self.minZ, self.maxZ), zLim=(self.minY,self.maxY))                             
                self.dots, = self.ax.plot([],[],[],'b.')            
                self.ani = animation.FuncAnimation(self.fig, self.drawDots)                
                plt.show()
                
                # Create gif movie
                outputFileName = "{}-{}-{}.gif"
                outputFileName.format(bvhObject.fileName,draw,skinning)
                print('Writing ',outputFileName.format(bvhObject.fileName,draw,skinning))                
                self.ani.save(outputFileName.format(bvhObject.fileName,draw,skinning), writer=animation.PillowWriter(fps=30))
                                  
                
        if skinning == 'LBS':
            # Linear Blend Skinning 
            # Algorithm:
            # for each vertex in rest pose mesh
            #   Go through skinning weights
            #       Get bind pose (inverse of local>world rest pose for that bone)
            #       Get transform for that bone
            #       Apply bind pose, apply bone rest > world transpose
            #   Vertex = mean world space position            
            print('Linear Blend Skinning..')                      
                
            for frame in range(frameStart,frameEnd,frameStep):
                           
                self.jointTFormList = []
                self.getJointTFormList(bvhObject, frame) 
                 
                npVerticesX = np.zeros(objObject.numVerts)
                npVerticesY = np.zeros(objObject.numVerts)
                npVerticesZ = np.zeros(objObject.numVerts)
                        
                # Take current pose and apply to rest pose mesh
                for vertIter in range(objObject.numVerts):
                              
                    finalVert = np.zeros((3,1))
                    for boneIter in range(bvhObject.totalJoints):
                                                 
                        thisSkinWeight = skinWeights[vertIter,boneIter]   

                        thisVert = objObject.vertices[vertIter]
                        thisVert = [float(thisVert[i]) for i in range(len(thisVert))]
                        thisVert.append(float(1))                
                        thisVert = np.transpose(np.array(thisVert))
                        
                        if thisSkinWeight>0:                
                            tForm = np.matmul(self.jointTFormList[boneIter],bvhObject.bindTfrms[boneIter])
                            thisVert = np.matmul(tForm,thisVert)
                            finalVert[0] = finalVert[0] + (thisVert[0] * thisSkinWeight)
                            finalVert[1] = finalVert[1] + (thisVert[1] * thisSkinWeight)
                            finalVert[2] = finalVert[2] + (thisVert[2] * thisSkinWeight)
                                                
                    npVerticesX[vertIter] = float(finalVert[0])
                    npVerticesY[vertIter] = float(finalVert[1])
                    npVerticesZ[vertIter] = float(finalVert[2])
                
                # Store this mesh/vertex set
                self.allVertsX.append(npVerticesX)
                self.allVertsY.append(npVerticesY)
                self.allVertsZ.append(npVerticesZ)
            
            # Get min and max values for x, y and z axis
            self.getAnimationBounds()
            
            if draw == 'Poly':
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.set_xbound(self.minX,self.maxX)
                ax.set_ybound(self.minY,self.maxY)
                ax.set_zbound(self.minZ,self.maxZ)

                self.animationPreview = []

                for frame in range(frameStart,frameEnd,frameStep):

                    verts = []
                    for face in range(len(self.npFacesSplit)):
                        tF = self.npFacesSplit[face]                        
                        
                        vX = [self.allVertsX[frame][tF[0]], self.allVertsX[frame][tF[1]], self.allVertsX[frame][tF[2]]]
                        vY = [self.allVertsY[frame][tF[0]], self.allVertsY[frame][tF[1]], self.allVertsY[frame][tF[2]]]
                        vZ = [self.allVertsZ[frame][tF[0]], self.allVertsZ[frame][tF[1]], self.allVertsZ[frame][tF[2]]]
                        
                        verts.append(list(zip(vX,vZ,vY))) # flip Y and Z to match bvh axis
    
                    self.animationPreview.append(verts)
                
                self.polyObj = Poly3DCollection(self.animationPreview[0], linewidths=0.1)
                self.polyObj.set_edgecolor('cyan')

                ax.add_collection3d(self.polyObj)
                self.ani = animation.FuncAnimation(fig, self.drawPoly)
                plt.show()
                                
                # Create gif movie
                outputFileName = "{}-{}-{}.gif"
                outputFileName.format(bvhObject.fileName,draw,skinning)
                print('Writing ',outputFileName.format(bvhObject.fileName,draw,skinning))
                self.ani.save(outputFileName.format(bvhObject.fileName,draw,skinning), writer=animation.PillowWriter(fps=30))
                
            if draw == 'Dots':                    
                # Draw animation
                self.fig = plt.figure()
        
                # NB swapping Y and Z as Y is up and not Z
                self.ax = self.fig.add_subplot(projection="3d",xlim=(self.minX,self.maxX), ylim=(self.minZ, self.maxZ), zLim=(self.minY,self.maxY))                             
                self.dots, = self.ax.plot([],[],[],'b.')            
                self.ani = animation.FuncAnimation(self.fig, self.drawDots)
                plt.show()
                
                # Create gif movie
                outputFileName = "{}-{}-{}.gif"
                outputFileName.format(bvhObject.fileName,draw,skinning)
                print('Writing ',outputFileName.format(bvhObject.fileName,draw,skinning))
                self.ani.save(outputFileName.format(bvhObject.fileName,draw,skinning), writer=animation.PillowWriter(fps=30))


    def drawPoly(self,frame):   
        if frame> len(self.allVertsX):
            self.ani.event_source.stop()
            
        self.polyObj.set_verts(self.animationPreview[frame])
    
    def drawDots(self, frame):
            
        if frame> len(self.allVertsX):
            self.ani.event_source.stop()
                   
        self.dots.set_data(self.allVertsX[frame],self.allVertsZ[frame])
        self.dots.set_3d_properties(self.allVertsY[frame])

        
    def getJointTFormList(self, bvhObject, frame):
       
        self.jointTFormList.append(bvhObject.root.transMats[frame])
                       
        # If there are children to the root, then start recursion to read the hierarchy
        if len(bvhObject.root.childNodes) > 0:
            for i in range(len(bvhObject.root.childNodes)):
                self.getJointTForm(bvhObject.root.childNodes[i], frame)

    def getJointTForm(self, currentNode, frame):
        if len(currentNode.childNodes) > 0:            
            self.jointTFormList.append(currentNode.transMats[frame])
            
            for i in range(len(currentNode.childNodes)):
                self.getJointTForm(currentNode.childNodes[i], frame)
            else:
                return

        else:
            self.jointTFormList.append(currentNode.transMats[frame])
   
    def getAnimationBounds(self):
        # Get min and max values for x, y and z axis        
        for iter in range(len(self.allVertsX)):
            thisMinX = min(self.allVertsX[iter])
            thisMaxX = max(self.allVertsX[iter])                
            thisMinY = min(self.allVertsY[iter])
            thisMaxY = max(self.allVertsY[iter])                
            thisMinZ = min(self.allVertsZ[iter])
            thisMaxZ = max(self.allVertsZ[iter])                
            if(thisMinX < self.minX):
                self.minX = thisMinX                    
            if(thisMaxX > self.maxX):
                self.maxX = thisMaxX                
            if(thisMinY < self.minY):
                self.minY = thisMinY                    
            if(thisMaxY > self.maxY):
                self.maxY = thisMaxY                
            if(thisMinZ < self.minZ):
                self.minZ = thisMinZ                    
            if(thisMaxZ > self.maxZ):
                self.maxZ = thisMaxZ
    
    def bvhGetJoints(self, bvhObject, frameStep=1):
          
        rootNode = bvhObject.root
        frame = 0
        frameStart = 0
        frameEnd = bvhObject.totalFrames              
        
        # Recursively read the 'pre estimated absolte Node coords' 
        # from the BVH Object and store parent to children connections creating a bone list.
        # This makes for easier drawing. NB - to see how to estimate abs coords see bvhRead func.
        for frame in range(frameStart,frameEnd,frameStep):
            currentJointCoords = rootNode.jointCoords[frame]            
            
            # If there are children to the root, then start recursion to read the hierarchy
            if len(rootNode.childNodes) > 0:
                for i in range(len(rootNode.childNodes)):                    
                    self.bvhGetBone(bvhObject, rootNode.childNodes[i], currentJointCoords, frame)               
        

    def bvhGetBone(self, bvhObject, currentNode, lastJointCoords, frame):        
        
        # Put all the bones in a list recursively to make drawing easier
        # While there are children to process, do one, then recurse to process further down the hierarchy
        if len(currentNode.childNodes) > 0:
                       
            currentJoint = currentNode.jointCoords[frame]   
            bvhObject.animationPreview.append([[lastJointCoords[0],currentJoint[0]],[lastJointCoords[1],currentJoint[1]],[lastJointCoords[2],currentJoint[2]]])                        
            
            for i in range(len(currentNode.childNodes)):
                self.bvhGetBone(bvhObject, currentNode.childNodes[i], currentJoint, frame)
            else:
                return
        else:            
            currentJoint = currentNode.jointCoords[frame]
            bvhObject.animationPreview.append([[lastJointCoords[0],currentJoint[0]],[lastJointCoords[1],currentJoint[1]],[lastJointCoords[2],currentJoint[2]]])            
    
    
    
    def makeTransMat(self, axisAngles, transOffsets, channelNames=['Xrotation','Yrotation','Zrotation']):
                
        # Make a composite rotation matrix from axis angles x,y,z
        # NB! for BVH files, very important that rotation concat follows
        # order specified in channelNames for the current Node 
        
        # Get channel ordering + abbreviate
        if(len(channelNames)==3):
            xRotPos = channelNames.index('Xrotation')
            yRotPos = channelNames.index('Yrotation')
            zRotPos = channelNames.index('Zrotation')
        elif(len(channelNames)==6):
            xRotPos = channelNames.index('Xrotation')-3
            yRotPos = channelNames.index('Yrotation')-3
            zRotPos = channelNames.index('Zrotation')-3       
        else:
            xRotPos=0
            yRotPos=1
            zRotPos=2
        
        xRad = math.radians(axisAngles[xRotPos])
        yRad = math.radians(axisAngles[yRotPos])
        zRad = math.radians(axisAngles[zRotPos])        
        
        Rx = np.zeros((3,3))
        Ry = np.zeros((3,3))
        Rz = np.zeros((3,3))
        transform = np.zeros((4,4))
                
        Rx[0,0] = 1
        Rx[1,1] = math.cos(xRad)
        Rx[1,2] = - math.sin(xRad)
        Rx[2,1] = math.sin(xRad)
        Rx[2,2] = math.cos(xRad)
        
        Ry[0,0] = math.cos(yRad)
        Ry[0,2] = math.sin(yRad)
        Ry[1,1] = 1
        Ry[2,0] = - math.sin(yRad)
        Ry[2,2] = math.cos(yRad)
        
        Rz[0,0] = math.cos(zRad)
        Rz[0,1] = - math.sin(zRad)
        Rz[1,0] = math.sin(zRad)
        Rz[1,1] = math.cos(zRad)
        Rz[2,2] = 1        
        
        # Apply rotations in correct order                
        #x,y,z
        if(xRotPos==0) and (yRotPos==1) and (zRotPos==2):
            transform[:3,:3] = np.matmul(Rx,np.matmul(Ry,Rz))
        
        #x,z,y
        if(xRotPos==0) and (yRotPos==2) and (zRotPos==1):
            transform[:3,:3] = np.matmul(Rx,np.matmul(Rz,Ry))
        
        #y,x,z
        if(xRotPos==1) and (yRotPos==0) and (zRotPos==2):
            transform[:3,:3] = np.matmul(Ry,np.matmul(Rx,Rz))
        
        #y,z,x
        if(xRotPos==2) and (yRotPos==0) and (zRotPos==1):
            transform[:3,:3] = np.matmul(Ry,np.matmul(Rz,Rx))
        
        #z,x,y
        if(xRotPos==1) and (yRotPos==2) and (zRotPos==0):
            transform[:3,:3] = np.matmul(Rz,np.matmul(Rx,Ry))
        
        #z,y,x
        if(xRotPos==2) and (yRotPos==1) and (zRotPos==0):
            transform[:3,:3] = np.matmul(Rz,np.matmul(Ry,Rx))
                        
        # Add translation
        transform[3,3] = 1    
        transform[0:3,3] = transOffsets
        
        return transform

         


if __name__ == '__main__':
    print('SkinData \'main\' is running the default demo..')
    print('Run as a program, this will run through basic usage.')
    print('System inputs', sys.argv)
    
    skinningFileName = 'skinningWeights.mat'
    skinObject = SkinData()
    skinObject.loadSkin(skinningFileName)
    
    bvhFileName = 'skeleton_motion_jump.bvh'    
    bvhObject = bvh.BVHData()
    bvhObject.bindPoseFrame = 147
    bvhObject.bvhRead(bvhFileName)
    skinObject.bvhGetJoints(bvhObject)
    
    objFileName = 'neutralMesh.obj'
    objObject = obj.OBJData()
    objObject.objRead(objFileName)
    
    skinObject.animateSkin(bvhObject,objObject,1,'RS','Dots') #LBS or RS, Dots or Poly
  
