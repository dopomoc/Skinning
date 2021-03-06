
SkinData Class - v1.0
Utility class for Skinning BVH Files
Written by Darren Cosker 2020

Requires
- OBJ object (and OBJData module)
- BVH object (and BVHData module)
- Skinning weights - IMPORTANT! Assumes loaded as a .mat array (Matlab). Easy to modify - just make self.skinWeights a verts X joints array
    
Given BVH and OBJ can perform Rigid Skinning or Linear Blend Skinning (LBS)

Example of Linear Blend Skinning (LBS) created by the class:

![Poly-LBS](https://github.com/dopomoc/Skinning/blob/master/skeleton_motion_jump.bvh-Poly-LBS.gif)
