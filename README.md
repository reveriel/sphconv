# Usage

## install

```
pip install .
```

## test

- Run output and gradient checks on the code by running `python check.py `.


## prepare dataset

same as in [second.pytorch](https://github.com/nutonomy/second.pytorch)

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Note: PointPillar's protos use ```KITTI_DATASET_ROOT=/data/sets/kitti_second/```.

#### 2. Create kitti infos:

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

#### 3. Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```


### Coordinate

RangeVoxel

```
    x  ▲          z
       │       /
       │      /
       │     /
┌────┐ │    /   ┌────┐
│ H  │ │   /    │ D  │
└────┘ │  /     └────┘
       │ /
       │/
       *──────────▶
                     y
           ┌────┐
           │ W  │
           └────┘
```

lidar coordinate

```
        z  ▲          x
           │       /
           │      /
           │     /
           │    /
           │   /
           │  /
           │ /
           │/
◀──────────*
  y
```

## reference

- [Peter Goldsborough](https://github.com/goldsborough) for pytorch extension
- [second.pytorch](https://github.com/nutonomy/second.pytorch/)
- [Custom ops for torchscript](https://brsoff.github.io/tutorials/advanced/torch_script_custom_ops.html#extending-torchscript-with-custom-c-operators)
- [Pytorch Extension with a Makefile](https://chrischoy.github.io/research/pytorch-extension-with-makefile/)


## TODO

- [ ] Tiled
- [x] to_dense, optimize
- [ ] multi GPU support

