'''
For each point cloud, labels consist of 3d bounding boxes of objects with
little intra-class shape variation
(all tree trunks in a point cloud are of similar width and point density,
bushes are of similar point density, etc)

We want to avoid augmenting data in any way that would suppress a valid,
learnable trait of the target classes

For example, we should not:
    - Move individual objects from their locations: many objects like trunks
        have learnable dependencies to other objects with specific traits
    - Rotate the point cloud or rotate individual objects:
        it is useful information that trees always stem from the same xy plane,
        and that the plane is always flat on the grid.
        This extends to other objects as well


Proposed Algorithm:

To maintain maximum information within classes across point clouds,
I propose the following augmentations, parts of which are adapted from
https://arxiv.org/pdf/2007.13373.pdf :

Given N objects in class C, voxelize based on class C's voxelization protocol:
    - tree trunks are voxelized in Z only, giving vertically stacked
      partitions, since the shape of the cylinder is repeated along the Z axis
    - bushes and other objects are voxelized in X, Y, and Z, since the shape
      of those objects is independent of axis

For each class C, choose 2 objects in class C, X and Y. As long as X and Y are
similar enough in shape, (for tree trunks, their widths should be similar,
for most objects, they are always similar),
swap V random partitions, where V is randomly sampled from [Vmin, Vmax]

Exception: for the bottom-most voxel and the top-most voxel in a tree trunk,
only swap with other bottom-most and top-most voxels respectively, to maintain
valid, learnable information about a tree trunks interaction with other objects

For each swap, before inserting the voxel from one object into the other,
rotate the voxel a random theta, and add 3D Guassian noise
(with a fixed standard deviation across swaps)
'''
