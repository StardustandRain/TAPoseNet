# TAPoseNet
[MICCAI 2024] TAPoseNet: Teeth Alignment based on Pose  estimation via multi-scale Graph Convolutional  Network
## Getting Started
You can run the given data in "dataset_test" directory. If you test your own data, please make sure that you have segmented the oral scan and ensure that the naming format of each tooth file matches the test data and that the overall coordinate system orientation of the teeth aligns with the test data.
### Prerequisites
The project is based on python3.7, torch==1.10.2, the packages below are also needed

```
trimesh, vtk, open3d
```
### Running the tests
You need to estimate the pose of teeth first. The pose of teeth are save in .npy file
```
python generate_pose.py
```
With the estimated pose of teeth, you can predict the teeth alignment target
```
python test.py
```
