# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pytransform3d import rotations as pr
# from pytransform3d import transformations as pt
# from pytransform3d.transform_manager import TransformManager
#
# ##### Get the P_1002 RFoot_Back_To_RightHeelCalibrated
# coordinate_1002_file_path = 'C:\\Users\\yicha\\Desktop\\Vicon_Data_In_Vicon_Coordinate_P_1002.csv'
# coordinate_1002_dataframe = pd.read_csv(coordinate_1002_file_path, sep=';')
#
# np_ground_to_RFoot_Back = np.array([
#     coordinate_1002_dataframe.at[0, 'RFoot_Back_Position_x'],
#     coordinate_1002_dataframe.at[0, 'RFoot_Back_Position_y'],
#     coordinate_1002_dataframe.at[0, 'RFoot_Back_Position_z'],
#     coordinate_1002_dataframe.at[0, 'RFoot_Back_Quaternion_w'],
#     coordinate_1002_dataframe.at[0, 'RFoot_Back_Quaternion_x'],
#     coordinate_1002_dataframe.at[0, 'RFoot_Back_Quaternion_y'],
#     coordinate_1002_dataframe.at[0, 'RFoot_Back_Quaternion_z']
# ])
#
# transform_ground_to_RFoot_Back = pt.transform_from_pq(np_ground_to_RFoot_Back)
#
# # print(pt.pq_from_transform(transform_ground_to_RFoot_Back))
#
# np_ground_to_RightHeelCalibrated = np.array([
#     coordinate_1002_dataframe.at[0, 'RightHeelCalibrated_Position_x'],
#     coordinate_1002_dataframe.at[0, 'RightHeelCalibrated_Position_y'],
#     coordinate_1002_dataframe.at[0, 'RightHeelCalibrated_Position_z'],
#     coordinate_1002_dataframe.at[0, 'RightHeelCalibrated_Quaternion_w'],
#     coordinate_1002_dataframe.at[0, 'RightHeelCalibrated_Quaternion_x'],
#     coordinate_1002_dataframe.at[0, 'RightHeelCalibrated_Quaternion_y'],
#     coordinate_1002_dataframe.at[0, 'RightHeelCalibrated_Quaternion_z']
# ])
#
# transform_ground_to_RightHeelCalibrated = pt.transform_from_pq(np_ground_to_RightHeelCalibrated)
#
# transform_manager = TransformManager()
#
# transform_manager.add_transform("ground", "RFoot_Back_1002", transform_ground_to_RFoot_Back)
# transform_manager.add_transform("ground", "RightHeelCalibrated_1002", transform_ground_to_RightHeelCalibrated)
#
# transform_RFoot_Back_to_RightHeelCalibrated = transform_manager.get_transform("RFoot_Back_1002", "RightHeelCalibrated_1002")
#
# ##### Get the P_1002 LFoot_Back_To_LeftHeelCalibrated
# coordinate_1002_file_path = 'C:\\Users\\yicha\\Desktop\\Vicon_Data_In_Vicon_Coordinate_P_1002.csv'
# coordinate_1002_dataframe = pd.read_csv(coordinate_1002_file_path, sep=';')
#
# np_ground_to_LFoot_Back = np.array([
#     coordinate_1002_dataframe.at[0, 'LFoot_Back_Position_x'],
#     coordinate_1002_dataframe.at[0, 'LFoot_Back_Position_y'],
#     coordinate_1002_dataframe.at[0, 'LFoot_Back_Position_z'],
#     coordinate_1002_dataframe.at[0, 'LFoot_Back_Quaternion_w'],
#     coordinate_1002_dataframe.at[0, 'LFoot_Back_Quaternion_x'],
#     coordinate_1002_dataframe.at[0, 'LFoot_Back_Quaternion_y'],
#     coordinate_1002_dataframe.at[0, 'LFoot_Back_Quaternion_z']
# ])
#
# transform_ground_to_LFoot_Back = pt.transform_from_pq(np_ground_to_LFoot_Back)
#
# # print(pt.pq_from_transform(transform_ground_to_LFoot_Back))
#
# np_ground_to_LeftHeelCalibrated = np.array([
#     coordinate_1002_dataframe.at[0, 'LeftHeelCalibrated_Position_x'],
#     coordinate_1002_dataframe.at[0, 'LeftHeelCalibrated_Position_y'],
#     coordinate_1002_dataframe.at[0, 'LeftHeelCalibrated_Position_z'],
#     coordinate_1002_dataframe.at[0, 'LeftHeelCalibrated_Quaternion_w'],
#     coordinate_1002_dataframe.at[0, 'LeftHeelCalibrated_Quaternion_x'],
#     coordinate_1002_dataframe.at[0, 'LeftHeelCalibrated_Quaternion_y'],
#     coordinate_1002_dataframe.at[0, 'LeftHeelCalibrated_Quaternion_z']
# ])
#
# transform_ground_to_LeftHeelCalibrated = pt.transform_from_pq(np_ground_to_LeftHeelCalibrated)
#
# transform_manager = TransformManager()
#
# transform_manager.add_transform("ground", "LFoot_Back_1002", transform_ground_to_LFoot_Back)
# transform_manager.add_transform("ground", "LeftHeelCalibrated_1002", transform_ground_to_LeftHeelCalibrated)
#
# transform_LFoot_Back_to_LeftHeelCalibrated = transform_manager.get_transform("LFoot_Back_1002", "LeftHeelCalibrated_1002")
#
# # print(transform_RFoot_Back_to_RightHeelCalibrated)
#
# ##### Fix the data for the subject P_1026
# folder_path_session = 'C:\\Users\\yicha\\Desktop\\Backup\\P_1026_Session_Complex'
# # folder_path_session = 'C:\\Users\\yicha\\Desktop\\Backup\\P_1026_Session_Crossing'
#
# lists_dir = os.listdir(folder_path_session + '\\Data')
# # print(lists_dir)
#
# for trial_dir in lists_dir:
#     csv_data_path = folder_path_session + '\\Data\\' + trial_dir + '\\Vicon\\Vicon_Data_In_Vicon_Coordinate.csv'
#     print(csv_data_path)
#
#     coordinate_1026_dataframe = pd.read_csv(csv_data_path, sep=';')
#
#     list_transform_RFoot_Back = []
#
#     for index, row in coordinate_1026_dataframe.iterrows():
#         np_ground_to_RFoot_Back = np.array([
#             row['RFoot_Back_Position_x'],
#             row['RFoot_Back_Position_y'],
#             row['RFoot_Back_Position_z'],
#             row['RFoot_Back_Quaternion_w'],
#             row['RFoot_Back_Quaternion_x'],
#             row['RFoot_Back_Quaternion_y'],
#             row['RFoot_Back_Quaternion_z']
#         ])
#
#         # print(np_ground_to_RFoot_Back)
#         if not np.isnan(np_ground_to_RFoot_Back).any():
#             transform_ground_to_RFoot_Back = pt.transform_from_pq(np_ground_to_RFoot_Back)
#
#             transform_manager = TransformManager()
#
#             transform_manager.add_transform("ground", "RFoot_Back_1026", transform_ground_to_RFoot_Back)
#             transform_manager.add_transform("RFoot_Back_1026", "RightHeelCalibrated_1026",
#                                             transform_RFoot_Back_to_RightHeelCalibrated)
#
#             transform_ground_to_RightHeelCalibrated = transform_manager.get_transform("ground", "RightHeelCalibrated_1026")
#
#             np_RightHeelCalibrated = pt.pq_from_transform(transform_ground_to_RightHeelCalibrated)
#
#             # print(np_RightHeelCalibrated)
#
#             coordinate_1026_dataframe.loc[index, 'RightHeelCalibrated_Position_x'] = np_RightHeelCalibrated[0]
#             coordinate_1026_dataframe.loc[index, 'RightHeelCalibrated_Position_y'] = np_RightHeelCalibrated[1]
#             coordinate_1026_dataframe.loc[index, 'RightHeelCalibrated_Position_z'] = np_RightHeelCalibrated[2]
#             coordinate_1026_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_w'] = np_RightHeelCalibrated[3]
#             coordinate_1026_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_x'] = np_RightHeelCalibrated[4]
#             coordinate_1026_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_y'] = np_RightHeelCalibrated[5]
#             coordinate_1026_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_z'] = np_RightHeelCalibrated[6]
#
#         np_ground_to_LFoot_Back = np.array([
#             row['LFoot_Back_Position_x'],
#             row['LFoot_Back_Position_y'],
#             row['LFoot_Back_Position_z'],
#             row['LFoot_Back_Quaternion_w'],
#             row['LFoot_Back_Quaternion_x'],
#             row['LFoot_Back_Quaternion_y'],
#             row['LFoot_Back_Quaternion_z']
#         ])
#
#         # print(np_ground_to_LFoot_Back)
#         if not np.isnan(np_ground_to_LFoot_Back).any():
#             transform_ground_to_LFoot_Back = pt.transform_from_pq(np_ground_to_LFoot_Back)
#
#             transform_manager = TransformManager()
#
#             transform_manager.add_transform("ground", "LFoot_Back_1026", transform_ground_to_LFoot_Back)
#             transform_manager.add_transform("LFoot_Back_1026", "LeftHeelCalibrated_1026",
#                                             transform_LFoot_Back_to_LeftHeelCalibrated)
#
#             transform_ground_to_LeftHeelCalibrated = transform_manager.get_transform("ground", "LeftHeelCalibrated_1026")
#
#             np_LeftHeelCalibrated = pt.pq_from_transform(transform_ground_to_LeftHeelCalibrated)
#
#             # print(np_LeftHeelCalibrated)
#
#             coordinate_1026_dataframe.loc[index, 'LeftHeelCalibrated_Position_x'] = np_LeftHeelCalibrated[0]
#             coordinate_1026_dataframe.loc[index, 'LeftHeelCalibrated_Position_y'] = np_LeftHeelCalibrated[1]
#             coordinate_1026_dataframe.loc[index, 'LeftHeelCalibrated_Position_z'] = np_LeftHeelCalibrated[2]
#             coordinate_1026_dataframe.loc[index, 'LeftHeelCalibrated_Quaternion_w'] = np_LeftHeelCalibrated[3]
#             coordinate_1026_dataframe.loc[index, 'LeftHeelCalibrated_Quaternion_x'] = np_LeftHeelCalibrated[4]
#             coordinate_1026_dataframe.loc[index, 'LeftHeelCalibrated_Quaternion_y'] = np_LeftHeelCalibrated[5]
#             coordinate_1026_dataframe.loc[index, 'LeftHeelCalibrated_Quaternion_z'] = np_LeftHeelCalibrated[6]
#
#     coordinate_1026_dataframe.to_csv(csv_data_path, sep=';')

########################################################################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

##### Get the P_1013 RFoot_Back_To_RightHeelCalibrated
coordinate_1013_file_path = 'C:\\Users\\yicha\\Desktop\\Vicon_Data_In_Vicon_Coordinate_P_1013.csv'
coordinate_1013_dataframe = pd.read_csv(coordinate_1013_file_path, sep=';')


np_ground_to_RFoot_Back = np.array([
    coordinate_1013_dataframe.at[0, 'RFoot_Back_Position_x'],
    coordinate_1013_dataframe.at[0, 'RFoot_Back_Position_y'],
    coordinate_1013_dataframe.at[0, 'RFoot_Back_Position_z'],
    coordinate_1013_dataframe.at[0, 'RFoot_Back_Quaternion_w'],
    coordinate_1013_dataframe.at[0, 'RFoot_Back_Quaternion_x'],
    coordinate_1013_dataframe.at[0, 'RFoot_Back_Quaternion_y'],
    coordinate_1013_dataframe.at[0, 'RFoot_Back_Quaternion_z']
])

transform_ground_to_RFoot_Back = pt.transform_from_pq(np_ground_to_RFoot_Back)

# print(pt.pq_from_transform(transform_ground_to_RFoot_Back))

np_ground_to_RightHeelCalibrated = np.array([
    coordinate_1013_dataframe.at[0, 'RightHeelCalibrated_Position_x'],
    coordinate_1013_dataframe.at[0, 'RightHeelCalibrated_Position_y'],
    coordinate_1013_dataframe.at[0, 'RightHeelCalibrated_Position_z'],
    coordinate_1013_dataframe.at[0, 'RightHeelCalibrated_Quaternion_w'],
    coordinate_1013_dataframe.at[0, 'RightHeelCalibrated_Quaternion_x'],
    coordinate_1013_dataframe.at[0, 'RightHeelCalibrated_Quaternion_y'],
    coordinate_1013_dataframe.at[0, 'RightHeelCalibrated_Quaternion_z']
])

transform_ground_to_RightHeelCalibrated = pt.transform_from_pq(np_ground_to_RightHeelCalibrated)

transform_manager = TransformManager()

transform_manager.add_transform("ground", "RFoot_Back_1013", transform_ground_to_RFoot_Back)
transform_manager.add_transform("ground", "RightHeelCalibrated_1013", transform_ground_to_RightHeelCalibrated)

transform_RFoot_Back_to_RightHeelCalibrated = transform_manager.get_transform("RFoot_Back_1013", "RightHeelCalibrated_1013")

# print(transform_RFoot_Back_to_RightHeelCalibrated)


##### Get the P_1013 LFoot_Front_To_LeftToeCalibrated
np_ground_to_LFoot_Front = np.array([
    coordinate_1013_dataframe.at[0, 'LFoot_Front_Position_x'],
    coordinate_1013_dataframe.at[0, 'LFoot_Front_Position_y'],
    coordinate_1013_dataframe.at[0, 'LFoot_Front_Position_z'],
    coordinate_1013_dataframe.at[0, 'LFoot_Front_Quaternion_w'],
    coordinate_1013_dataframe.at[0, 'LFoot_Front_Quaternion_x'],
    coordinate_1013_dataframe.at[0, 'LFoot_Front_Quaternion_y'],
    coordinate_1013_dataframe.at[0, 'LFoot_Front_Quaternion_z']
])

transform_ground_to_LFoot_Front = pt.transform_from_pq(np_ground_to_LFoot_Front)

# print(pt.pq_from_transform(transform_ground_to_RFoot_Back))

np_ground_to_LeftToeCalibrated = np.array([
    coordinate_1013_dataframe.at[0, 'LeftToeCalibrated_Position_x'],
    coordinate_1013_dataframe.at[0, 'LeftToeCalibrated_Position_y'],
    coordinate_1013_dataframe.at[0, 'LeftToeCalibrated_Position_z'],
    coordinate_1013_dataframe.at[0, 'LeftToeCalibrated_Quaternion_w'],
    coordinate_1013_dataframe.at[0, 'LeftToeCalibrated_Quaternion_x'],
    coordinate_1013_dataframe.at[0, 'LeftToeCalibrated_Quaternion_y'],
    coordinate_1013_dataframe.at[0, 'LeftToeCalibrated_Quaternion_z']
])

transform_ground_to_LeftToeCalibrated = pt.transform_from_pq(np_ground_to_LeftToeCalibrated)

transform_manager = TransformManager()

transform_manager.add_transform("ground", "LFoot_Front_1013", transform_ground_to_LFoot_Front)
transform_manager.add_transform("ground", "LeftToeCalibrated_1013", transform_ground_to_LeftToeCalibrated)

transform_LFoot_Front_to_LeftToeCalibrated = transform_manager.get_transform("LFoot_Front_1013", "LeftToeCalibrated_1013")


##### Fix the data for the subject P_2006
folder_path_session = 'C:\\Users\\yicha\\Desktop\\Backup\\P_2006_Session_Complex'
# folder_path_session = 'C:\\Users\\yicha\\Desktop\\Backup\\P_2006_Session_Crossing'

lists_dir = os.listdir(folder_path_session + '\\Data')
# print(lists_dir)

for trial_dir in lists_dir:
    csv_data_path = folder_path_session + '\\Data\\' + trial_dir + '\\Vicon\\Vicon_Data_In_Vicon_Coordinate.csv'
    print(csv_data_path)

    coordinate_2006_dataframe = pd.read_csv(csv_data_path, sep=';')

    list_transform_RFoot_Back = []

    for index, row in coordinate_2006_dataframe.iterrows():

        np_ground_to_RFoot_Back = np.array([
            row['RFoot_Back_Position_x'],
            row['RFoot_Back_Position_y'],
            row['RFoot_Back_Position_z'],
            row['RFoot_Back_Quaternion_w'],
            row['RFoot_Back_Quaternion_x'],
            row['RFoot_Back_Quaternion_y'],
            row['RFoot_Back_Quaternion_z']
        ])

        # print(np_ground_to_RFoot_Back)
        if not np.isnan(np_ground_to_RFoot_Back).any():
            transform_ground_to_RFoot_Back = pt.transform_from_pq(np_ground_to_RFoot_Back)

            transform_manager = TransformManager()

            transform_manager.add_transform("ground", "RFoot_Back_2006", transform_ground_to_RFoot_Back)
            transform_manager.add_transform("RFoot_Back_2006", "RightHeelCalibrated_2006", transform_RFoot_Back_to_RightHeelCalibrated)

            transform_ground_to_RightHeelCalibrated = transform_manager.get_transform("ground", "RightHeelCalibrated_2006")

            np_RightHeelCalibrated = pt.pq_from_transform(transform_ground_to_RightHeelCalibrated)

            # print(np_RightHeelCalibrated)

            coordinate_2006_dataframe.loc[index, 'RightHeelCalibrated_Position_x'] = np_RightHeelCalibrated[0]
            coordinate_2006_dataframe.loc[index, 'RightHeelCalibrated_Position_y'] = np_RightHeelCalibrated[1]
            coordinate_2006_dataframe.loc[index, 'RightHeelCalibrated_Position_z'] = np_RightHeelCalibrated[2]
            coordinate_2006_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_w'] = np_RightHeelCalibrated[3]
            coordinate_2006_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_x'] = np_RightHeelCalibrated[4]
            coordinate_2006_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_y'] = np_RightHeelCalibrated[5]
            coordinate_2006_dataframe.loc[index, 'RightHeelCalibrated_Quaternion_z'] = np_RightHeelCalibrated[6]

        np_ground_to_LFoot_Front = np.array([
            row['LFoot_Front_Position_x'],
            row['LFoot_Front_Position_y'],
            row['LFoot_Front_Position_z'],
            row['LFoot_Front_Quaternion_w'],
            row['LFoot_Front_Quaternion_x'],
            row['LFoot_Front_Quaternion_y'],
            row['LFoot_Front_Quaternion_z'],
        ])

        if not np.isnan(np_ground_to_LFoot_Front).any():
            transform_ground_to_LFoot_Front = pt.transform_from_pq(np_ground_to_LFoot_Front)

            transform_manager = TransformManager()

            transform_manager.add_transform("ground", "LFoot_Front_2006", transform_ground_to_LFoot_Front)
            transform_manager.add_transform("LFoot_Front_2006", "LeftToeCalibrated_2006",
                                            transform_LFoot_Front_to_LeftToeCalibrated)

            transform_ground_to_LeftToeCalibrated = transform_manager.get_transform("ground", "LeftToeCalibrated_2006")

            np_LeftToeCalibrated = pt.pq_from_transform(transform_ground_to_LeftToeCalibrated)

            # print(np_RightHeelCalibrated)

            coordinate_2006_dataframe.loc[index, 'LeftToeCalibrated_Position_x'] = np_LeftToeCalibrated[0]
            coordinate_2006_dataframe.loc[index, 'LeftToeCalibrated_Position_y'] = np_LeftToeCalibrated[1]
            coordinate_2006_dataframe.loc[index, 'LeftToeCalibrated_Position_z'] = np_LeftToeCalibrated[2]
            coordinate_2006_dataframe.loc[index, 'LeftToeCalibrated_Quaternion_w'] = np_LeftToeCalibrated[3]
            coordinate_2006_dataframe.loc[index, 'LeftToeCalibrated_Quaternion_x'] = np_LeftToeCalibrated[4]
            coordinate_2006_dataframe.loc[index, 'LeftToeCalibrated_Quaternion_y'] = np_LeftToeCalibrated[5]
            coordinate_2006_dataframe.loc[index, 'LeftToeCalibrated_Quaternion_z'] = np_LeftToeCalibrated[6]

    coordinate_2006_dataframe.to_csv(csv_data_path, sep=';')
