# VoxelNet PyTorch

Use the VoxelNet architecture as described [here](https://arxiv.org/abs/1711.06396) to predict KITTI objects using lidar.
This will be expanded to predict obkects in dense forest point clouds, like tree trunks soon.

## Table of Contents
- [Transforms](#Transforms)
  - [Pointcloud Conversion](#Pointcloud-Conversion)
  - [Targets](#Targets)
  - [Post Processing](#Post-Processing)
- [Usage](#Usage)
- [Visualization](#Visualization)

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
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;B is the max number of points per voxel (if there are more than B points, randomly sample B points to keep),  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; *See 2.1.1: Feature Learning Network: Random  Sampling*  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;7 encodes
<code>[x, y, z, r, x-v<sub>x</sub>, y-v<sub>y</sub>, z-v<sub>z</sub>]</code>, where `v` is the centroid of the points in the voxel, and `r` is the reflectance.  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; *See 2.1.1: Feature Learning Network: Stacked Voxel Feature Encoding*  
<br /><br />
From this voxelization, we also get&ensp;&ensp;&ensp;`(X, 3)` ⇾ `(N, 3)`

where  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;N is the number of non-empty voxels,  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;4 encodes `[X_Voxel_Index, Y_Voxel_Index, Z_Voxel_Index]` 

This array is useful for converting the voxel array from sparse to dense format, and from dense to sparse format, since it tells us where the non-empty voxels belong.

Once we have these two arrays, we can pass them through the VoxelNet architecture, as described in *2.1 VoxelNet Architecture*

---

### Targets
In order to calculate loss, we have to calculate a few arrays: we'll call them `anchors`, `pos_equal_one`, `neg_equal_one` and `targets`.

---

Our `anchors` are essentially the options for the network for where it thinks the objects are. For VoxelNet, we have 70,400 anchors.
To [calculate these anchors](https://github.com/aaronzberger/CMU_VoxelNet/blob/8fb81d1eb2a1855ab2ca9947cc8d4bfe70c0aebd/src/utils.py#L89-L125),
we simply make a grid and place anchors every x, every y, and every z, at specified rotations (where x, y, and z are constants). The anchors are in the form `[x, y, z, height, width, length, rotation]`

---

The VoxelNet architecture will output two arrays:  
&ensp;&ensp;&ensp;probability score map: the probability (0 to 1) of each anchor being an object  
&ensp;&ensp;&ensp;regression map: the distance from the predicted object to the anchor  

Our `pos_equal_one` array is the ground truth for the probability score map.
There are 70,400 values, each either 0 or 1. If there are X ground truth labels, the array will contain X 1s and all other values will be zero

Our `neg_equal_one` array is simply the exact opposite of `pos_equal_one`, containing 1s everywhere except where there are objects.

---

Lastly, our `targets` array is the ground truth for the regression map. This array has 70,400 arrays, each encoding 

&ensp;&ensp;&ensp;`[Δx, Δy, Δz, Δl, Δw, Δh, Δθ]`, where these deltas are calculated according to *2.2 Loss Function*


### Post Processing
After VoxelNet outputs a probability score map and a regression map, as outlined in the section above, we have to convert these values to bounding boxes.

This process is not described in the paper, but we can simply obtain x, y, z, l, w, h, θ by using the inverse of how we calculated the `targets` array above.
See [here](https://github.com/aaronzberger/CMU_VoxelNet/blob/7f730eacae1f024400f4b245f0b241321bdf8e07/src/conversions.py#L325-L391) to see how we calculate these values. 

This method is validated: If we pass in the `pos_equal_one` and `targets` array, which are the ground truths for the probability score map and regression map,
the bounding boxes match the ground truths specified by the original labels.

## Usage
First, in [`config.py`](https://github.com/aaronzberger/CMU_VoxelNet/blob/7f730eacae1f024400f4b245f0b241321bdf8e07/src/config.py), chance `base_dir` to
the path to your `CMU_VoxelNet` folder. Change `data_dir` to the path to your KITTI dataset directory.  

To run training, use

`python main.py train`

Specify hyperparameters in the [`config.json`](https://github.com/aaronzberger/CMU_VoxelNet/blob/7f730eacae1f024400f4b245f0b241321bdf8e07/config.json) file.

The training module saves the visualizations of X examples every E epochs, where X is the `num_viz` variable, and E is the `viz_every` variable in the `config.json` file.

## Visualization
Once you've run the training module, your `CMU_VoxelNet` folder will have a `viz` folder inside. This folder will contain files named like `epochXXXpclXXXXXX.npz`.

These are the saved prediction bounding boxes, lidar, and ground truth boxes. To display them. Use the `viz_3d.py` file.

To visualize every file in the viz folder in order: `python viz_3d.py all`.

To visualize specific files: `python viz_3d.py these [PATH_TO_FILE1] [PATH_TO_FILE2] ...`

When you visualize, the __green boxes are the ground truths__ and the __red boxes are the predictions__.

The terminal will also display how many ground truth and prediction boxes are being displayed.
