import argparse
import json
import math
import os
import sys

# add the repository root directory to the python path
from src.utils.detection import Detection
from src.utils.utils import get_2d_corner_points, id_to_class_name_mapping, class_name_to_id_mapping

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src", ".."))

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("sys.path: ", sys.path)

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R

import numpy as np

from src.utils.vis_utils import VisualizationUtils
import internal.src.hd_map.hd_map as hdmap

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--output_file_path_bev_plot",
        type=str,
        help="Output file path for bev plot.",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_train",
        type=str,
        help="Path to train labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_val",
        type=str,
        help="Path to val labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sampled",
        type=str,
        help="Path to test sampled labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_test_sequence",
        type=str,
        help="Path to test sequence labels",
        default="",
    )

    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_south",
        type=str,
        help="Path to r01_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s01_north",
        type=str,
        help="Path to r02_s01 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_south",
        type=str,
        help="Path to r02_s02 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s02_north",
        type=str,
        help="Path to r02_s02 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_south",
        type=str,
        help="Path to r02_s03 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s03_north",
        type=str,
        help="Path to r02_s03 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_south",
        type=str,
        help="Path to r02_s04 south lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--input_folder_path_labels_sequence_s04_north",
        type=str,
        help="Path to r02_s04 north lidar labels",
        default="",
    )
    arg_parser.add_argument(
        "--output_folder_path_statistic_plots",
        type=str,
        help="output folder path to statistics plots",
        default="",
    )
    args = arg_parser.parse_args()
    output_folder_path_statistic_plots = args.output_folder_path_statistic_plots

    input_folder_paths_all = []
    if args.input_folder_path_labels_train:
        input_folder_paths_all.append(args.input_folder_path_labels_train)
    if args.input_folder_path_labels_val:
        input_folder_paths_all.append(args.input_folder_path_labels_val)
    if args.input_folder_path_labels_test_sampled:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sampled)
    if args.input_folder_path_labels_test_sequence:
        input_folder_paths_all.append(args.input_folder_path_labels_test_sequence)

    if args.input_folder_path_labels_sequence_s01_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_south)
    if args.input_folder_path_labels_sequence_s01_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s01_north)
    if args.input_folder_path_labels_sequence_s02_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_south)
    if args.input_folder_path_labels_sequence_s02_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s02_north)
    if args.input_folder_path_labels_sequence_s03_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_south)
    if args.input_folder_path_labels_sequence_s03_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s03_north)
    if args.input_folder_path_labels_sequence_s04_south:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_south)
    if args.input_folder_path_labels_sequence_s04_north:
        input_folder_paths_all.append(args.input_folder_path_labels_sequence_s04_north)

    bev_fig = None
    bev_ax = None
    # TODO: do not hardcode sequence names
    # boxes_list = {
    #     "r02_s01": {},
    #     "r02_s02": {},
    #     "r02_s03": {},
    #     "r02_s04": {},
    # }
    boxes_list = {
        "r03_s01": {},
        "r03_s02": {},
        "r03_s03": {},
        "r03_s04": {},
        "r03_s05": {},
        "r03_s06": {},
        "r03_s07": {},
        "r03_s08": {},
    }
    lane_sections = hdmap.load_map_for_local_frame("TrafficX_base")
    # filter lane section to 200 m x 200 m region around TrafficX base
    lane_sections_filtered = []
    for lane_section in lane_sections:
        lane_section = lane_section.crop_to_area(min_pos=np.array([-100, -100]), max_pos=np.array([100, 100]))
        if lane_section:
            lane_sections_filtered.append(lane_section)

    classes = [
        "CAR",
        "TRUCK",
        "TRAILER",
        "VAN",
        "MOTORCYCLE",
        "BUS",
        "PEDESTRIAN",
        "BICYCLE",
        "EMERGENCY_VEHICLE",
        "OTHER",
    ]
    classes_valid_set = set()
    valid_ids = set()

    for input_folder_path_labels in input_folder_paths_all:
        input_files_labels = sorted(os.listdir(input_folder_path_labels))
        # TODO: do not hardcode sequence names
        if "r02_s01" in input_folder_path_labels:
            sequence = "r02_s01"
        elif "r02_s02" in input_folder_path_labels:
            sequence = "r02_s02"
        elif "r02_s03" in input_folder_path_labels:
            sequence = "r02_s03"
        elif "r02_s04" in input_folder_path_labels:
            sequence = "r02_s04"
        elif "r03_s01" in input_folder_path_labels:
            sequence = "r03_s01"
        elif "r03_s02" in input_folder_path_labels:
            sequence = "r03_s02"
        elif "r03_s03" in input_folder_path_labels:
            sequence = "r03_s03"
        elif "r03_s04" in input_folder_path_labels:
            sequence = "r03_s04"
        elif "r03_s05" in input_folder_path_labels:
            sequence = "r03_s05"
        elif "r03_s06" in input_folder_path_labels:
            sequence = "r03_s06"
        elif "r03_s07" in input_folder_path_labels:
            sequence = "r03_s07"
        elif "r03_s08" in input_folder_path_labels:
            sequence = "r03_s08"
        else:
            sequence = "r03_s01"
            # raise ValueError("Unknown dataset type")
        for label_file_name in input_files_labels:
            json_file = open(
                os.path.join(input_folder_path_labels, label_file_name),
            )
            json_data = json.load(json_file)
            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                num_labeled_objects = len(frame_obj["objects"].keys())
                for object_track_id, object_json in frame_obj["objects"].items():
                    object_class = object_json["object_data"]["type"]

                    classes_valid_set.add(object_class)
                    valid_ids.add(classes.index(object_class))

                    if "cuboid" in object_json["object_data"]:
                        cuboid = object_json["object_data"]["cuboid"]["val"]
                        location = cuboid[0:3]
                        quaternion = np.asarray(cuboid[3:7])
                        roll, pitch, yaw = R.from_quat(quaternion).as_euler("xyz", degrees=False)
                        track_history_attribute = VisualizationUtils.get_attribute_by_name(
                            object_json["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                        )
                        if boxes_list[sequence].get(label_file_name) is None:
                            boxes_list[sequence][label_file_name] = []
                        boxes_list[sequence][label_file_name].append(
                            Detection(
                                location=location,
                                dimensions=(cuboid[7], cuboid[8], cuboid[9]),
                                yaw=yaw,
                                category=object_class,
                                pos_history=track_history_attribute["val"],
                                uuid=object_track_id,
                            )
                        )

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
        }
    )
    bev_fig, bev_ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    plt.subplots_adjust(left=0.1, right=1.0, top=0.98, bottom=0.12)
    bev_ax.set_aspect("equal")
    bev_ax.set_xlim(-50, 50)
    bev_ax.set_ylim(0, 100)

    # plot all lane sections from hd map
    for lane_section in lane_sections_filtered:
        for lane in lane_section.lanes:
            bev_ax.plot(lane[:, 0], lane[:, 1], color=(0.3, 0.3, 0.3), linewidth=1.0, zorder=0)


    # remove not valid classes
    classes_valid_list = list(classes_valid_set)
    for class_name in classes.copy():
        if class_name not in classes_valid_list:
            classes.remove(class_name)

    # set legend for plot using class names
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=id_to_class_name_mapping[str(class_name_to_id_mapping[class_name])]["color_rgb_normalized"],
            lw=4,
            label=class_name if class_name != "EMERGENCY_VEHICLE" else "EMERGENCY_VEH",
        )
        for class_name in classes
    ]

    # plot legend with black edge color
    bev_ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
        frameon=True,
        edgecolor="black",
    )
    bev_ax.set_xlabel("Longitude [m]", fontsize=14)
    bev_ax.set_ylabel("Latitude [m]", fontsize=14)
    bev_ax.tick_params(axis="both", which="major", labelsize=14)
    bev_ax.tick_params(axis="both", which="minor", labelsize=12)

    # iterate all boxes
    framd_idx = 0
    for sequence, dataset_objects in boxes_list.items():
        for label_file_name, boxes in dataset_objects.items():
            for box in boxes:
                class_id = class_name_to_id_mapping[box.category.upper()]
                class_color_rgb_normalized = id_to_class_name_mapping[str(class_id)]["color_rgb_normalized"]
                # plot black rectangle using location and rotation of detection

                # fill rectangle with color
                draw_box = False
                if draw_box:
                    px, py = get_2d_corner_points(
                        box.location[0], box.location[1], box.dimensions[0], box.dimensions[1], box.yaw - math.pi
                    )
                    bev_ax.fill(px, py, color=class_color_rgb_normalized, alpha=0.5, zorder=2)
                    bev_ax.plot(px, py, color="black", linewidth=1.0, zorder=3)
                draw_arrow = False
                if draw_arrow:
                    bev_ax.arrow(
                        box.location[0],
                        box.location[1],
                        box.dimensions[0] * 0.6 * np.cos((box.yaw + math.pi) % (2 * math.pi)),
                        box.dimensions[0] * 0.6 * np.sin((box.yaw + math.pi) % (2 * math.pi)),
                        head_width=box.dimensions[1] * 0.3,
                        color="k",
                    )

                # plot track
                if len(box.pos_history) > 0:
                    # set z-order to 0.5 to make sure that the track is plotted behind the rectangle
                    # (otherwise the rectangle would be hidden by the track)
                    locations = np.reshape(box.pos_history, (-1, 3))

                    # rotate location by 80 degree
                    # locations_rotated = np.zeros_like(locations)
                    # locations_rotated[:, 0] = locations[:, 0] * np.cos(78.5 * np.pi / 180) - locations[:, 1] * np.sin(
                    #     78.5 * np.pi / 180
                    # )
                    # locations_rotated[:, 1] = locations[:, 0] * np.sin(78.5 * np.pi / 180) + locations[:, 1] * np.cos(
                    #     78.5 * np.pi / 180
                    # )
                    # # translate location by -10 m in x direction
                    # locations_rotated[:, 0] = locations_rotated[:, 0] - 15
                    # # translate location by 1 m in y direction
                    # locations_rotated[:, 1] = locations_rotated[:, 1] + 2

                    # TODO: check whether south or north is loaded

                    transformation_lidar_to_base = None
                    rotation_lidar_to_base = None
                    if "TrafficX_lidar_ouster_south" in label_file_name:
                        # rotation_lidar_south_to_base = 85.5
                        rotation_lidar_south_to_base = 0.0
                        rotation_lidar_to_base = rotation_lidar_south_to_base
                        # transform from lidar south coordinate frame to TrafficX base coordinate frame
                        y_offset = 1.3  # 4.81100903
                        # transformation_lidar_south_to_base = np.array(
                        #     [
                        #         [0.99011437, -0.13753536, -0.02752358, y_offset],
                        #         [0.13828977, 0.99000475, 0.02768645, -15.94524012],
                        #         [0.02344061, -0.03121898, 0.99923766, -8.05225519],
                        #         [0.0, 0.0, 0.0, 1.0],
                        #     ]
                        # )
                        # working
                        # transformation_lidar_south_to_base = np.array(
                        #     [
                        #         [0.21479486, -0.97610280, 0.03296187, -15.81918059],
                        #         [0.97627128, 0.21553835, 0.02091894, 2.33407956],
                        #         [-0.02752358, 0.02768645, 0.99923767, 6.58394667],
                        #         [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                        #     ]
                        # )
                        transformation_lidar_south_to_base = np.array(
                            [
                                [0.21479485, -0.9761028, 0.03296187, -15.87257873],
                                [0.97627128, 0.21553835, 0.02091894, 2.30019086],
                                [-0.02752358, 0.02768645, 0.99923767, 7.48077521],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                        transformation_lidar_to_base = transformation_lidar_south_to_base
                    elif "TrafficX_lidar_ouster_north" in label_file_name:
                        # rotation_lidar_north_to_base = 77.5
                        rotation_lidar_north_to_base = 0.0
                        rotation_lidar_to_base = rotation_lidar_north_to_base
                        x_offset = -1.0  # 0.45151759
                        y_offset = -2.0  # -3.00306778
                        # transformation_lidar_north_to_base = np.array(
                        #     [
                        #         [0.95902442, 0.28247922, -0.02185554, x_offset],
                        #         [-0.28180664, 0.95901925, 0.02944626, y_offset],
                        #         [0.02927784, -0.02208065, 0.9993274, -8.52743828],
                        #         [0.0, 0.0, 0.0, 1.0],
                        #     ]
                        # )
                        # working
                        # transformation_lidar_north_to_base = np.array(
                        #     [
                        #         [-0.06821837, -0.99735900, 0.02492560, -2.02963586],
                        #         [0.99751775, -0.06774959, 0.01919179, 0.56416412],
                        #         [-0.01745241, 0.02617296, 0.99950507, 7.00000000],
                        #         [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                        #     ]
                        # )
                        # from calibration with bohan
                        transformation_lidar_north_to_base = np.array(
                            [
                                [-0.064419, -0.997922, 0.00169282, -2.08748],
                                [0.997875, -0.0644324, -0.00969147, 0.226579],
                                [0.0097804, 0.0010649, 0.999952, 8.29723],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                        transformation_lidar_to_base = transformation_lidar_north_to_base
                    else:
                        raise ValueError("Unknown lidar sensor ID in file name.")

                    locations = np.hstack((locations, np.ones((locations.shape[0], 1))))
                    locations_transformed = np.matmul(transformation_lidar_to_base, locations.T).T

                    # rotate location by 80 degree
                    locations_rotated = np.zeros_like(locations_transformed)
                    locations_rotated[:, 0] = locations_transformed[:, 0] * np.cos(
                        rotation_lidar_to_base * np.pi / 180
                    ) - locations_transformed[:, 1] * np.sin(rotation_lidar_to_base * np.pi / 180)
                    locations_rotated[:, 1] = locations_transformed[:, 0] * np.sin(
                        rotation_lidar_to_base * np.pi / 180
                    ) + locations_transformed[:, 1] * np.cos(rotation_lidar_to_base * np.pi / 180)

                    bev_ax.plot(
                        np.array(locations_rotated)[:, 0],
                        np.array(locations_rotated)[:, 1],
                        color=class_color_rgb_normalized,
                        zorder=1,
                    )
                else:
                    print(f"Warning: No history for {box.id} {box.category}", flush=True)
            # TODO: for debugging save every plot
            bev_fig.savefig(
                args.output_file_path_bev_plot.replace(".pdf", "_")
                + str(framd_idx).zfill(9)
                + "_"
                + str(label_file_name.replace(".json", ""))
                + "_"
                + sequence
                + ".jpg",
            )
            framd_idx += 1

    print(f"Saving BEV plot to {args.output_file_path_bev_plot}", flush=True)
    bev_fig.savefig(os.path.join(output_folder_path_statistic_plots, "bev_plot_all_drives.pdf"))
    plt.close()
    plt.clf()
