# TraffiX Dataset Development Kit

The TraffiX (`TraffiX`) Dataset is based on roadside sensor data. The dataset includes anonymized and precision-timestamped multi-modal sensor and object data in high resolution, covering a variety of traffic situations. We provide camera and LiDAR frames from overhead gantry bridges with the corresponding objects labeled with 3D bounding boxes. The dataset contains the following subsets:
- TraffiX Dataset (`TraffiX`)
- TraffiX Intersection Dataset (`TraffiX-I`)
- TraffiX Cooperative Dataset (`TraffiX-C`) 



The Development Kit provides a dataset loader for images, point clouds, labels and calibration data. The calibration loader reads the intrinsic and extrinsic calibration information. The projection matrix is then used to visualize the 2D and 3D labels on cameras images. 

## Installation
Create an anaconda environment:
```
conda create --name TraffiX-dataset-dev-kit python=3.9
conda activate TraffiX-dataset-dev-kit
```
Install the following dependencies:
```
conda install -c conda-forge fvcore
conda install -c conda-forge iopath
```
In case, you are using NVIDIA CUDA <11.6:
```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar -xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
```
Install PyTorch3D:
```
pip3 install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1121/download.html
```

Install requirements:

```
pip3 install -r requirements.txt
pip3 install --upgrade git+https://github.com/klintan/pypcd.git
```


## Dataset Structure
#### 1) TraffiX  Dataset (`TraffiX`) 
The TraffiX Dataset (`TraffiX`) contains 5 subsets (`s00` to `s04`) and is structured in the following way:

The first 3 sets `TraffiX_r00_s00`, `TraffiX_r00_s01` and `TraffiX_r00_s02` contain image data (`.png`) from roadside cameras with corresponding label files (stored in OpenLABEL `.json` format) and calibration data:
``` 
├── TraffiX_dataset_r00_s00
│   ├── _images
│   │   ├── s040_camera_basler_north_16mm
│   │   ├── s040_camera_basler_north_50mm
│   │   ├── s050_camera_basler_south_16mm
│   │   ├── s050_camera_basler_south_50mm
│   ├── _labels
│   │   ├── s040_camera_basler_north_16mm
│   │   ├── s040_camera_basler_north_50mm
│   │   ├── s050_camera_basler_south_16mm
│   │   ├── s050_camera_basler_south_50mm
│   ├── _calibration
│   │   ├── s040_camera_basler_north_16mm.json
│   │   ├── s040_camera_basler_north_50mm.json
│   │   ├── s050_camera_basler_south_16mm.json
│   │   ├── s050_camera_basler_south_50mm.json
```

The last two sets `TraffiXc_r00_s03`, and `TraffiX_r00_s04` contain point cloud data (`.pcd`) from roadside LiDARs with corresponding label files (stored in OpenLABEL `.json` format) and calibration data:
``` 
├── TraffiX_dataset_r00_s03
│   ├── _points
│   ├── _labels
```


#### 2) TraffiX Dataset (`TraffiX`) extension with sequences from real accidents
The `TraffiX_r01` dataset contains 3 subsets (`s01` to `s03`) and is structured in the following way:

Example: TraffiX_dataset_r01_s01:
``` 
├── TraffiX_dataset_r01_s03
│   ├── _images
│   │   ├── s040_camera_basler_north_16mm
│   │   ├── s040_camera_basler_north_50mm
│   │   ├── s050_camera_basler_south_16mm
│   │   ├── s050_camera_basler_south_50mm
│   ├── _labels
│   │   ├── s040_camera_basler_north_16mm
│   │   ├── s040_camera_basler_north_50mm
│   │   ├── s050_camera_basler_south_16mm
│   │   ├── s050_camera_basler_south_50mm
│   ├── _calibration
│   │   ├── s040_camera_basler_north_16mm.json
│   │   ├── s040_camera_basler_north_50mm.json
│   │   ├── s050_camera_basler_south_16mm.json
│   │   ├── s050_camera_basler_south_50mm.json
```

#### 3) TraffiX Intersection Dataset (`TraffiX-I`)
The TraffiX Intersection Dataset (`TraffiX-I`) contains 4 subsets (`s01` to `s04`) and is structured in the following way:

``` 
├── TraffiX_intersection_dataset_r02_s01
│   ├── _images
│   │   ├── TrafficX_camera_basler_south1_8mm
│   │   ├── TrafficX_camera_basler_south2_8mm
│   ├── _labels
│   │   ├── TrafficX_lidar_ouster_south
│   │   ├── TrafficX_lidar_ouster_north
│   ├── _points_clouds
│   │   ├── TrafficX_lidar_ouster_south
│   │   ├── TrafficX_lidar_ouster_north
├── TraffiX_intersection_dataset_r02_s02
│   ├── ...
├── TraffiX_intersection_dataset_r02_s03
│   ├── ...
├── TraffiX_intersection_dataset_r02_s04
│   ├── ...
```

## 1. Label Visualization
### 1.1 Visualization of labels in camera images 
The following visualization script can be used to draw the 2D and/or 3D labels on camera frames:

```
python TraffiX-dataset-dev-kit/src/visualization/visualize_image_with_3d_boxes.py --camera_id TrafficX_camera_basler_south1_8mm \
                                                                                      --lidar_id TrafficX_lidar_ouster_south \
                                                                                      --input_folder_path_images <IMAGE_FOLDER_PATH> \
                                                                                      --input_folder_path_point_clouds <POINT_CLOUD_FOLDER_PATH> \
                                                                                      --input_folder_path_labels <LABEL_FOLDER_PATH> \
                                                                                      --viz_mode [box2d,box3d,point_cloud,track_history] \
                                                                                      --viz_color_mode [by_category,by_track_id] \
                                                                                      --output_folder_path_visualization <OUTPUT_FOLDER_PATH>
```
 
![labeling_example](./img/camera_basler_south2_8mm.jpg)

### 1.2 Visualization of labels in LiDAR point cloud scans
The script below draws labels on a LiDAR frame:

```
python TraffiX-dataset-dev-kit/src/visualization/visualize_point_cloud_with_labels.py --input_folder_path_point_clouds <INPUT_FOLDER_PATH_POINT_CLOUDS> \
                                                                                          --input_folder_path_labels <INPUT_FOLDER_PATH_LABELS> \
                                                                                          --save_visualization_results \
                                                                                          --output_folder_path_visualization_results <OUTPUT_FOLDER_PATH_VISUALIZATION_RESULTS>
```


## 2. Image Undistortion/Rectification
The development kit also contains a script to undistort camera images:

```
python TraffiX-dataset-dev-kit/src/preprocessing/undistort_images.py --input_folder_path_images ~TraffiX_dataset_r00_s00/_images/s040_camera_basler_north_16mm \
                                                                         --file_path_calibration_parameter ~/TraffiX_dataset_r00_s00/_calibration/s40_camera_basler_north_16mm.json \
                                                                         --output_folder_path_images ~/TraffiX_dataset_r00_s00/_images_undistorted
```
An example between a distorted an undistorted image is shown below:
![undistortion_example](./img/undistortion_example.gif)

## 3. Point Cloud Pre-Processing

### 3.1 Point Cloud Registration

The following script can be used to register point clouds from two different LiDARs:
```
python TraffiX-dataset-dev-kit/src/preprocessing/register_point_clouds.py --folder_path_point_cloud_source <INPUT_FOLDER_PATH_POINT_CLOUDS_SOURCE> \
                                                             --folder_path_point_cloud_target <INPUT_FOLDER_PATH_POINT_CLOUDS_TARGET> \
                                                             --save_registered_point_clouds \
                                                             --output_folder_path_registered_point_clouds <OUTPUT_FOLDER_PATH_POINT_CLOUDS>
```
![registered_point_cloud](./img/registered_point_cloud.png)

### 3.2 Noise Removal
A LiDAR preprocessing module reduces noise in point cloud scans:

```
python TraffiX-dataset-dev-kit/src/preprocessing/remove_noise_from_point_clouds.py --input_folder_path_point_clouds <INPUT_FOLDER_PATH_POINT_CLOUDS> \
                                                                                       --output_folder_path_point_clouds <OUTPUT_FOLDER_PATH_POINT_CLOUDS>
```
![noise_removal](./img/outlier_removal.png)

## 4. Label Conversion
In addition, a data converter/exporter enables you to convert the labels from OpenLABEL format into other formats like KITTI, COCO or YOLO and the other way round. 


### OpenLABEL to YOLO
The following script converts the OpenLABEL labels into YOLO labels:
```
python TraffiX-dataset-dev-kit/src/converter/conversion_openlabel_to_yolo.py --input_folder_path_labels <INPUT_FOLDER_PATH_LABELS> \
                                                                                 --output_folder_path_labels <OUTPUT_FOLDER_PATH_LABELS>
```


## 5. Data Split
The script below splits the dataset into `train` and `val`:

```
python TraffiX-dataset-dev-kit/src/preprocessing/create_train_val_split.py --input_folder_path_dataset <INPUT_FOLDER_PATH_DATASET> \
                                                                               --input_folder_path_data_split_root <INPUT_FOLDER_PATH_DATA_SPLIT_ROOT>
```


## 6. Evaluation Script
Finally, a model evaluation script is provided to benchmark your models on the TraffiX-Dataset.
```
python TraffiX-dataset-dev-kit/src/eval/evaluation.py --folder_path_ground_truth ${FILE/DIR} \
                                                          --folder_path_predictions ${FILE/DIR} \  
                                                          [--object_min_points ${NUM}]
```
Dataformat of predictions - one TXT file per frame with the content (one line per predicted object): class x y z l w h rotation_z.<br>
Example
```
Car 16.0162 -28.9316 -6.45308 2.21032 3.74579 1.18687 2.75634
Car 17.926 -19.4624 -7.0266 1.03365 0.97037 0.435425 0.82854
```
Example call to compare one ground truth file with one prediction file visually:
```
python TraffiX-dataset-dev-kit/src/eval/evaluation.py --folder_path_ground_truth ~/TraffiX_dataset_r01_test/labels/1651673050_454284855_TrafficX_lidar_ouster_south.json \
                                                          --folder_path_predictions ~/predictions/1651673050_454284855_TrafficX_lidar_ouster_south.json \ 
                                                          --object_min_points 0
```
Example call to evaluate the whole set if ground truth bounding boxes enclose more than 20 points:
```
python TraffiX-dataset-dev-kit/src/eval/evaluation.py --folder_path_ground_truth ~/TraffiX_dataset_r01_test_set/labels \
                                                          --folder_path_predictions ~/detections \
                                                          --object_min_points 20
```
Final result when evaluating the TraffiX-Dataset R1 test set vs. itself:
```

|AP@50             |overall     |Occurrence (pred/gt)|
|Vehicle           |100.00      |2110/2110           |
|Pedestrian        |100.00      |32/32               |
|Bicycle           |100.00      |156/156             |
|mAP               |100.00      |2298/2298 (Total)   |
```

# License

The TraffiX Dataset Development Kit scripts are released under MIT license as found in the license file.
