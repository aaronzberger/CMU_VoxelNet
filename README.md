# VoxelNet PyTorch

Use the VoxelNet architecture as described [here](https://arxiv.org/abs/1711.06396) to predict KITTI objects using lidar.
This will be expanded to predict obkects in dense forest point clouds, like tree trunks soon.

## Table of Contents
- [Transforms](#Transforms)
  - [Input](#Input)
  - [Targets](#Targets)
  - [Post Processing](#Post-Processing)
- [Usage](#Usage)
- [Visualization](#Visualization)
- [Data Augmentation](#Data-Augmentation)

## Transforms
There are a number of conversions we must perform to fully extract bounding box predictions from a raw pointcloud.

### Pointcloud Conversion
First, we [load the pointcloud](https://github.com/aaronzberger/CMU_VoxelNet/blob/8fb81d1eb2a1855ab2ca9947cc8d4bfe70c0aebd/src/dataset.py#L223)
as a raw (X, 3) array, where X is the number of points.

---

Next, we [voxelize the pointcloud](https://github.com/aaronzberger/CMU_VoxelNet/blob/8fb81d1eb2a1855ab2ca9947cc8d4bfe70c0aebd/src/dataset.py#L162-L208):
&ensp;&ensp;&ensp;`(X, 3)` ⇾ `(A, B, 7)`

where  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;A is the number of voxels,  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;*See 2.1.1: Feature Learning Network: Voxel Partition*  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;B is the maximum number of points per voxel (if there are more than B points, randomly sample B points to keep),  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; *See 2.1.1: Feature Learning Network: Random  Sampling*  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;7 encodes
<code>[x, y, z, r, x-v<sub>x</sub>, y-v<sub>y</sub>, z-v<sub>z</sub>]</code>, where `v` is the centroid of the points in the voxel  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; *See 2.1.1: Feature Learning Network: Stacked Voxel Feature Encoding*  

From this voxelization, we also get&ensp;&ensp;&ensp;`(X, 3)` ⇾ `(N, 3)`

where  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;N is the number of non-empty voxels,  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;4 encodes `[X_Voxel_Index, Y_Voxel_Index, Z_Voxel_Index]` 

This array is useful for converting the voxel array from sparse to dense format, and from dense to sparse format, since it tells us where the non-empty voxels belong.

Once we have these two arrays, we can pass them through the VoxelNet architecture, as described in *2.1 VoxelNet Architecture*

---

### Targets
In order to calculate loss, we have to calculate a few arrays: we'll call them `anchors`, `targets`, `pos_equal_one`, and `neg_equal_one`.

---

Our `anchors` are essentially the options for the network for where it thinks the objects are. For VoxelNet, we have 70,400 anchors.
To [calculate these anchors](https://github.com/aaronzberger/CMU_VoxelNet/blob/8fb81d1eb2a1855ab2ca9947cc8d4bfe70c0aebd/src/utils.py#L89-L125),
we simply make a grid and place anchors every x, every y, and every z (where x, y, and z are constants). The anchors are in the form `[x, y, z, h, w, l, r]`

---

The VoxelNet architecture will output two arrays:  
&ensp;&ensp;&ensp;probability score map: the probability (0 to 1) of each anchor being an object  
&ensp;&ensp;&ensp;regression map: the distance from the predicted object to the anchor  




### Post Processing

## Usage
To run the trainin
