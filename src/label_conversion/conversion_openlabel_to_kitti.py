from glob import glob

import cv2
from pypcd import pypcd
from tqdm import tqdm
import argparse
import json
import ntpath
import numpy as np
import os.path
import shutil


# Module description:
# This module converts the OpenLABEL format to KITTI format.

# Requirements:
#   - The dataset needs to be in the OpenLABEL format.
#   - The dataset needs to be split into a training and validation set: src/preprocessing/create_train_val_split.py

# Usage:
#           python src/label_conversion/conversion_openlabel_to_kitti.py --root-dir dataset/r02_train_val_split \
#                                                                        --version point_cloud \
#                                                                        --out-dir dataset/r02_train_val_split_in_kitti \
#                                                                        --format num


class OpenLABEL2KITTIConverter(object):
    """OpenLABEL to KITTI label format converter.
       This class serves as the converter to change the OpenLABEL format to KITTI format.
    """

    def __init__(self, splits, root_dir, out_dir, file_name_format):
        """
        Args:
            splits list[(str)]: Contains the different splits
            root_dir (str): Input folder path to OpenLABEL labels.
            out_dir (str): Output folder path to save data in KITTI format.
            file_name_format (str): Output file name of the converted file
        """

        self.splits = splits
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.file_name_format = file_name_format

        self.map_version_to_dir = {
            'training': 'train',
            'validation': 'val',
            'testing': 'test'
        }

        self.map_traffix_to_kitti_sensors = {
            'TrafficX_lidar_ouster_north': 'velodyne_1',
            'TrafficX_lidar_ouster_south': 'velodyne_2',
            'TrafficX_camera_basler_south1_8mm': 'image_1',
            'TrafficX_camera_basler_south2_8mm': 'image_2'
        }

        self.train_set = []
        self.val_set = []
        self.test_set = []

        self.map_set_to_dir_idx = {
            'training': 0,
            'validation': 1,
            'testing': 2
        }

        self.imagesets = {
            'training': self.train_set,
            'validation': self.val_set,
            'testing': self.test_set
        }

        self.occlusion_map = {
            'NOT_OCCLUDED': 0,
            'PARTIALLY_OCCLUDED': 1,
            'MOSTLY_OCCLUDED': 2
        }

    def convert(self):

        print('Start converting ...')
        num_frames_training = -1
        for split in self.splits:
            for sensor_type in tqdm(
                    os.listdir(os.path.join(self.root_dir, self.map_version_to_dir[split], 'point_clouds'))):
                print(f'Converting split: {split}...')

                point_cloud_file_list = sorted(
                    glob(os.path.join(self.root_dir, self.map_version_to_dir[split], 'point_clouds', sensor_type, '*')))
                if split == "training":
                    num_frames_training = len(point_cloud_file_list)
                for file_idx, input_file_path_point_cloud in tqdm(enumerate(point_cloud_file_list)):
                    if split == "validation":
                        file_idx = file_idx + num_frames_training
                    out_file_name_no_ext = self.format_file_name(file_idx, input_file_path_point_cloud)
                    point_cloud_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/{self.map_traffix_to_kitti_sensors[sensor_type]}'
                    os.makedirs(point_cloud_out_dir, exist_ok=True)
                    self.save_point_cloud(input_file_path_point_cloud,
                                          os.path.join(point_cloud_out_dir, f'{out_file_name_no_ext}.bin'))

                    input_file_path_label = input_file_path_point_cloud.replace('point_clouds',
                                                                                'labels_point_clouds').replace('pcd',
                                                                                                               'json')

                    # load json labels
                    with open(input_file_path_label) as f:
                        label_json_data = json.load(f)

                    # parse calibration data from label files
                    # TODO: parse from label_json_data
                    projection_matrix_velo_1_to_cam_2 = np.array(
                        [[1318.95273325, -859.15213894, -289.13390611, 11272.03223502],
                         [90.01799314, -2.9727517, -1445.63809767, 585.78988153],
                         [0.876766, 0.344395, -0.335669, -7.26891]])
                    projection_matrix_velo_1_to_cam_1 = np.array()
                    projection_matrix_velo_2_to_cam_2 = np.array()
                    projection_matrix_velo_2_to_cam_1 = np.array()
                    r0_rect = np.eye(3)
                    tr_velo_1_to_cam_2 = np.array()
                    tr_velo_1_to_cam_1 = np.array()
                    tr_velo_2_to_cam_2 = np.array()
                    tr_velo_2_to_cam_1 = np.array()

                    # write calibration data to file
                    calib_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/calib'
                    os.makedirs(calib_out_dir, exist_ok=True)
                    self.save_calibration(os.path.join(calib_out_dir, f'{out_file_name_no_ext}.txt'),
                                          projection_matrix_velo_1_to_cam_2, projection_matrix_velo_1_to_cam_1,
                                          projection_matrix_velo_2_to_cam_2, projection_matrix_velo_2_to_cam_1,
                                          r0_rect, tr_velo_1_to_cam_2, tr_velo_1_to_cam_1, tr_velo_2_to_cam_2,
                                          tr_velo_2_to_cam_1)


                    if sensor_type == 'TrafficX_lidar_ouster_north':
                        label_dir_name = 'label_1'
                    elif sensor_type == 'TrafficX_lidar_ouster_south':
                        label_dir_name = 'label_2'
                    else:
                        raise ValueError(
                            'Sensor type not supported. Please choose between "TrafficX_lidar_ouster_north" or "TrafficX_lidar_ouster_south"')

                    label_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/{label_dir_name}'
                    os.makedirs(label_out_dir, exist_ok=True)
                    self.save_label(input_file_path_label, os.path.join(label_out_dir, f'{out_file_name_no_ext}.txt'))
                    self.imagesets[split].append(out_file_name_no_ext + '\n')
                    file_idx += 1
        for split in self.splits:
            for sensor_type in tqdm(os.listdir(os.path.join(self.root_dir, self.map_version_to_dir[split], 'images'))):
                image_file_list = sorted(
                    glob(os.path.join(self.root_dir, self.map_version_to_dir[split], 'images', sensor_type, '*')))
                for file_idx, input_file_path_image in tqdm(enumerate(image_file_list)):
                    if split == "validation":
                        file_idx = file_idx + num_frames_training
                    out_file_name_no_ext = self.format_file_name(file_idx, input_file_path_image)
                    image_out_dir = f'{self.out_dir}/{self.map_version_to_dir[split]}/{self.map_traffix_to_kitti_sensors[sensor_type]}'
                    os.makedirs(image_out_dir, exist_ok=True)
                    self.save_image(input_file_path_image, os.path.join(image_out_dir, out_file_name_no_ext + '.png'))

        print('Creating ImageSets...')
        self.create_imagesets()
        print('\nFinished ...')

    def format_file_name(self, file_idx, input_file_path):
        """
        Create the specified file name convention
        Args:
            file_idx: Index of the file in the given split
            input_file_path: Input file path

        Returns: Specified file name without extension

        """
        if self.file_name_format == 'name':
            return os.path.basename(input_file_path).split('.')[0]
        else:
            return f'{str(file_idx).zfill(6)}'

    def save_calib(self, file, out_file):
        # TODO: Add calibration file
        pass

    @staticmethod
    def save_point_cloud(input_file_path_point_cloud, output_file_path_point_cloud):
        """
        Converts file from .pcd to .bin
        Args:
            input_file_path_point_cloud: Input file path to .pcd file
            output_file_path_point_cloud: Output filepath to .bin file
        """
        point_cloud = pypcd.PointCloud.from_path(input_file_path_point_cloud)
        np_x = np.array(point_cloud.pc_data['x'], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data['y'], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data['z'], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data['intensity'], dtype=np.float32) / 256
        bin_format = np.column_stack((np_x, np_y, np_z, np_i))
        bin_format.tofile(os.path.join(output_file_path_point_cloud))

    def save_label(self, input_file_path_label, output_file_path_label):
        """
        Converts OpenLABEL format to KITTI label format
        Args:
            input_file_path_label: Input file path to .json label file
            output_file_path_label: Output file path to .txt label file
        """
        # read json file
        lines = []
        json_file = open(input_file_path_label)
        json_data = json.load(json_file)
        for frame_id, label_json in json_data['openlabel']['frames'].items():
            for track_uuid, label_object in label_json['objects'].items():
                category = label_object['object_data']['type']
                # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
                # TODO: use projected box to check for truncation
                truncated = 0.00
                # 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown
                occluded = 3
                for item in label_object['object_data']['cuboid']['attributes']['text']:
                    if item['name'] == 'occlusion_level':
                        occluded = self.occlusion_map[item['val']]
                # Observation angle of object, ranging [-pi..pi]
                # TODO: calculate observation angle
                alpha = 0.00
                cuboid = label_object['object_data']['cuboid']['val']
                x_center = cuboid[0]
                y_center = cuboid[1]
                z_center = cuboid[2]
                length = cuboid[7]
                width = cuboid[8]
                height = cuboid[9]
                _, _, yaw = self.quaternion_to_euler(cuboid[3], cuboid[4], cuboid[5], cuboid[6])
                # TODO: calculate 2D box from 3D box by projecting 3D box to image plane
                bounding_box = [
                    x_center - length / 2,
                    y_center - width / 2,
                    x_center + length / 2,
                    y_center + width / 2,
                ]
                line = f"{category} {round(truncated, 2)} {occluded} {round(alpha, 2)} " + \
                       f"{round(bounding_box[0], 2)} {round(bounding_box[1], 2)} {round(bounding_box[2], 2)} " + \
                       f"{round(bounding_box[3], 2)} {round(height, 2)} {round(width, 2)} {round(length, 2)} " + \
                       f"{round(x_center, 2)} {round(y_center, 2)} {round(z_center, 2)} {round(yaw, 2)}\n"
                lines.append(line)
        fp_label = open(output_file_path_label, 'a')
        fp_label.writelines(lines)
        fp_label.close()

    @staticmethod
    def quaternion_to_euler(q0, q1, q2, q3):
        """
        Converts quaternions to euler angles using unique transformation via atan2

        Returns: roll, pitch and yaw

        """
        roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
        pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
        yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        return roll, pitch, yaw

    def create_imagesets(self):
        """
        Creates the ImageSets train.txt, val.txt, trainval.txt and test.txt each containing corresponding files
        """
        os.makedirs(os.path.join(self.out_dir, 'ImageSets'))

        with open(os.path.join(self.out_dir, 'ImageSets', 'train.txt'), 'w') as file:
            file.writelines(self.train_set)

        with open(os.path.join(self.out_dir, 'ImageSets', 'val.txt'), 'w') as file:
            file.writelines(self.val_set)

        with open(os.path.join(self.out_dir, 'ImageSets', 'trainval.txt'), 'w') as file:
            file.writelines(self.train_set)
            file.writelines(self.val_set)

        with open(os.path.join(self.out_dir, 'ImageSets', 'test.txt'), 'w') as file:
            file.writelines(self.test_set)

    def save_image(self, input_file_path_image, output_file_path_image):
        """
        Saves image to new location as .png
        Args:
            input_file_path_image: Input file path to image
            output_file_path_image: Output file path to image
        """
        img = cv2.imread(input_file_path_image)
        cv2.imwrite(output_file_path_image, img)

    def save_calibration(self, param, projection_matrix_velo_1_to_cam_2, projection_matrix_velo_1_to_cam_1,
                         projection_matrix_velo_2_to_cam_2, projection_matrix_velo_2_to_cam_1, r0_rect,
                         tr_velo_1_to_cam_2, tr_velo_1_to_cam_1, tr_velo_2_to_cam_2, tr_velo_2_to_cam_1):
        """
        Saves calibration to new location as .txt
        :param param:
        :param projection_matrix_velo_1_to_cam_2:
        :param projection_matrix_velo_1_to_cam_1:
        :param projection_matrix_velo_2_to_cam_2:
        :param projection_matrix_velo_2_to_cam_1:
        :param r0_rect:
        :param tr_velo_1_to_cam_2:
        :param tr_velo_1_to_cam_1:
        :param tr_velo_2_to_cam_2:
        :param tr_velo_2_to_cam_1:
        :return:
        """
        lines = []
        lines.append('P0: ' + ' '.join(map(str, projection_matrix_velo_1_to_cam_2)) + '\n')
        lines.append('P1: ' + ' '.join(map(str, projection_matrix_velo_1_to_cam_1)) + '\n')
        lines.append('P2: ' + ' '.join(map(str, projection_matrix_velo_2_to_cam_2)) + '\n')
        lines.append('P3: ' + ' '.join(map(str, projection_matrix_velo_2_to_cam_1)) + '\n')
        lines.append('R0_rect: ' + ' '.join(map(str, r0_rect)) + '\n')
        lines.append('Tr_velo_to_cam_0: ' + ' '.join(map(str, tr_velo_1_to_cam_2)) + '\n')
        lines.append('Tr_velo_to_cam_1: ' + ' '.join(map(str, tr_velo_1_to_cam_1)) + '\n')
        lines.append('Tr_velo_to_cam_2: ' + ' '.join(map(str, tr_velo_2_to_cam_2)) + '\n')
        lines.append('Tr_velo_to_cam_3: ' + ' '.join(map(str, tr_velo_2_to_cam_1)) + '\n')
        fp_calib = open(os.path.join(self.out_dir, 'calib', param + '.txt'), 'w')
        fp_calib.writelines(lines)
        fp_calib.close()



def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-dir',
        type=str,
        default='.',
        help='Specify the root folder path to the dataset.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='.',
        required=False,
        help='name of info pkl')
    parser.add_argument(
        '--file-name-format',
        type=str,
        default='num',
        required=False,
        choices=['name', 'num'],
        help="""specify whether to keep original filenames or convert to numbering (e.g. 000000.txt) to be mmdetection3d compatible"""
    )
    parser.add_argument(
        '--pkl',
        default=False,
        action='store_true',
        required=False,
        help="""Create .pkl files for mmdetection3d"""
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    splits = ['training', 'validation']
    converter = OpenLABEL2KITTIConverter(
        splits=splits,
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        file_name_format=args.file_name_format
    )
    converter.convert()
    if args.pkl:
        # TODO: Create pkl file
        pass
    # create_openlabel_info_file(root_path, out_dir, info_prefix, workers)
    # create_groundtruth_database("openlabel", root_path, info_prefix, f'{out_dir}/{info_prefix}_infos_train.pkl')
