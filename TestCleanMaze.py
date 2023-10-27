import os
import glob
import shutil
import random
import numpy as np

# main_path_maze = 'D:\\JNJ\\Maze\\Version_clinical_study\\'
# 
# file_list = []
# directory_list = []
# final_directory_list = []
# potential_directory_list = []
# 
# for root, dirs, files in os.walk(main_path_maze, topdown=False):
# 
#     for name in files:
#         file_name = os.path.join(root, name)
#         file_list.append(file_name)
# 
#        # print(os.path.join(root, name))
#     for name in dirs:
#         directory_name = os.path.join(root, name)
#         directory_list.append(directory_name)
# 
#         if 'first_step_left' in directory_name or 'first_step_right' in directory_name:
#             if 'final' in directory_name:
#                 final_directory_list.append(directory_name)
#             else:
#                 potential_directory_list.append(directory_name)

# print(potential_directory_list)

# # Rearrange the maze image in the potential_directory_list
# for potential_directory in potential_directory_list:
#
#     print(potential_directory)
#
#     # Get the list of file potential to place in final
#     potential_file_list = []
#     for file_name in os.listdir(potential_directory):
#         if 'median' in file_name:
#             potential_file_list.append(potential_directory + '\\' + file_name)
#
#     for i in range(9, 16):
#
#         index_required_file_existed = False
#
#         for potential_file in potential_file_list:
#             # Get the index of the file
#             if 'extra' in potential_file:
#                 file_index = potential_file.split('_')[-2]
#             else:
#                 file_index = potential_file.split('_')[-1].split('.')[0]
#
#             # If the index equals to i, then copy them inside the final folder
#             if file_index == i:
#                 potential_file_name = potential_file.split('\\')[-1]
#
#                 if 'extra' in potential_file_name:
#                     list_string = potential_file_name.split('_')
#                     list_string[-2] = str(i)
#                     new_potential_file_name = '_'.join(list_string)
#
#                 elif 'csv' in potential_file_name:
#                     list_string = potential_file_name.split('_')
#                     list_string[-1] = str(i)
#                     new_potential_file_name = '_'.join(list_string) + '.csv'
#
#                 elif 'png' in potential_file_name:
#                     list_string = potential_file_name.split('_')
#                     list_string[-1] = str(i)
#                     new_potential_file_name = '_'.join(list_string) + '.png'
#
#                 potential_file_destination = potential_file.replace(potential_file_name, 'final\\' + new_potential_file_name)
#                 shutil.move(potential_file, potential_file_destination)
#                 potential_file_list.remove(potential_file)
#
#                 index_required_file_existed = True
#
#         # If the file existed for the required index, move the file
#         # Otherwise, grab a number, change the number to the required index and move the file
#         if not index_required_file_existed:
#
#             if len(potential_file_list) != 0:
#
#                 random_index = random.randrange(0, len(potential_file_list))
#                 # random_index = random.randrange(1, 50)
#                 # Get the index of the file
#                 if 'extra' in potential_file_list[random_index]:
#                     index_to_change = potential_file_list[random_index].split('_')[-2]
#                 else:
#                     index_to_change = potential_file_list[random_index].split('_')[-1].split('.')[0]
#
#                 for potential_file in potential_file_list:
#                     # Get the index of the file
#                     if 'extra' in potential_file:
#                         file_index = potential_file.split('_')[-2]
#                     else:
#                         file_index = potential_file.split('_')[-1].split('.')[0]
#
#                     if file_index == index_to_change:
#                         potential_file_name = potential_file.split('\\')[-1]
#
#                         if 'extra' in potential_file_name:
#                             list_string = potential_file_name.split('_')
#                             list_string[-2] = str(i)
#                             new_potential_file_name = '_'.join(list_string)
#
#                         elif 'csv' in potential_file_name:
#                             list_string = potential_file_name.split('_')
#                             list_string[-1] = str(i)
#                             new_potential_file_name = '_'.join(list_string) + '.csv'
#
#                         elif 'png' in potential_file_name:
#                             list_string = potential_file_name.split('_')
#                             list_string[-1] = str(i)
#                             new_potential_file_name = '_'.join(list_string) + '.png'
#
#                         potential_file_destination = potential_file.replace(potential_file_name, 'final\\' + new_potential_file_name)
#                         shutil.move(potential_file, potential_file_destination)
#                         potential_file_list.remove(potential_file)


# # Delete all the potential files
# for potential_directory in potential_directory_list:
# 
#     print(potential_directory)
# 
#     # Get the list of file potential to place in final
#     potential_file_list = []
#     for file_name in os.listdir(potential_directory):
#         if 'median' in file_name:
#             potential_file_list.append(potential_directory + '\\' + file_name)
# 
#     for potential_file in potential_file_list:
#         os.remove(potential_file)
# 
# # Move all file in final directory into the potential derectory
# for final_directory in final_directory_list:
# 
#     print(final_directory)
# 
#     final_file_list = []
#     for file_name in os.listdir(final_directory):
#         if 'median' in file_name:
#             final_file_list.append(final_directory + '\\' + file_name)
# 
#     for final_file in final_file_list:
#         final_file_destination = final_file.replace('\\final', '')
#         shutil.move(final_file, final_file_destination)
# 
# # Delete all final folder
# for final_directory in final_directory_list:
#     os.rmdir(final_directory)

import pandas as pd

exchange_folder_path = 'D:\\JNJ\\Exchange\\'
original_file_name = 'Vicon_Data_In_Vicon_Coordinate_Original.csv'
data_file_name = 'Vicon_Data_In_Vicon_Coordinate_Data.csv'
destination_file_name = 'Vicon_Data_In_Vicon_Coordinate.csv'

original_file_dataframe = pd.read_csv(exchange_folder_path + original_file_name, sep=';', index_col=[0])
data_file_dataframe = pd.read_csv(exchange_folder_path + data_file_name, sep=';', index_col=[0])

all_header = ['Step051_Occluded', 'Step051_Position_x', 'Step051_Position_y', 'Step051_Position_z', 'Step051_Quaternion_x', 'Step051_Quaternion_y', 'Step051_Quaternion_z', 'Step051_Quaternion_w', 'Step051_Marker_Step0511_Position_x', 'Step051_Marker_Step0511_Position_y', 'Step051_Marker_Step0511_Position_z', 'Step051_Marker_Step0512_Position_x', 'Step051_Marker_Step0512_Position_y', 'Step051_Marker_Step0512_Position_z', 'Step051_Marker_Step0513_Position_x', 'Step051_Marker_Step0513_Position_y', 'Step051_Marker_Step0513_Position_z', 'Step051_Marker_Step0514_Position_x', 'Step051_Marker_Step0514_Position_y', 'Step051_Marker_Step0514_Position_z', 'Step051_Marker_Step0515_Position_x', 'Step051_Marker_Step0515_Position_y', 'Step051_Marker_Step0515_Position_z', 'Step052_Occluded', 'Step052_Position_x', 'Step052_Position_y', 'Step052_Position_z', 'Step052_Quaternion_x', 'Step052_Quaternion_y', 'Step052_Quaternion_z', 'Step052_Quaternion_w', 'Step052_Marker_Step0521_Position_x', 'Step052_Marker_Step0521_Position_y', 'Step052_Marker_Step0521_Position_z', 'Step052_Marker_Step0522_Position_x', 'Step052_Marker_Step0522_Position_y', 'Step052_Marker_Step0522_Position_z', 'Step052_Marker_Step0523_Position_x', 'Step052_Marker_Step0523_Position_y', 'Step052_Marker_Step0523_Position_z', 'Step052_Marker_Step0524_Position_x', 'Step052_Marker_Step0524_Position_y', 'Step052_Marker_Step0524_Position_z', 'Step052_Marker_Step0525_Position_x', 'Step052_Marker_Step0525_Position_y', 'Step052_Marker_Step0525_Position_z', 'Step151_Occluded', 'Step151_Position_x', 'Step151_Position_y', 'Step151_Position_z', 'Step151_Quaternion_x', 'Step151_Quaternion_y', 'Step151_Quaternion_z', 'Step151_Quaternion_w', 'Step151_Marker_Step1511_Position_x', 'Step151_Marker_Step1511_Position_y', 'Step151_Marker_Step1511_Position_z', 'Step151_Marker_Step1512_Position_x', 'Step151_Marker_Step1512_Position_y', 'Step151_Marker_Step1512_Position_z', 'Step151_Marker_Step1513_Position_x', 'Step151_Marker_Step1513_Position_y', 'Step151_Marker_Step1513_Position_z', 'Step151_Marker_Step1514_Position_x', 'Step151_Marker_Step1514_Position_y', 'Step151_Marker_Step1514_Position_z', 'Step151_Marker_Step1515_Position_x', 'Step151_Marker_Step1515_Position_y', 'Step151_Marker_Step1515_Position_z', 'Step152_Occluded', 'Step152_Position_x', 'Step152_Position_y', 'Step152_Position_z', 'Step152_Quaternion_x', 'Step152_Quaternion_y', 'Step152_Quaternion_z', 'Step152_Quaternion_w', 'Step152_Marker_Step1521_Position_x', 'Step152_Marker_Step1521_Position_y', 'Step152_Marker_Step1521_Position_z', 'Step152_Marker_Step1522_Position_x', 'Step152_Marker_Step1522_Position_y', 'Step152_Marker_Step1522_Position_z', 'Step152_Marker_Step1523_Position_x', 'Step152_Marker_Step1523_Position_y', 'Step152_Marker_Step1523_Position_z', 'Step152_Marker_Step1524_Position_x', 'Step152_Marker_Step1524_Position_y', 'Step152_Marker_Step1524_Position_z', 'Step152_Marker_Step1525_Position_x', 'Step152_Marker_Step1525_Position_y', 'Step152_Marker_Step1525_Position_z']
destination_dataframe = pd.DataFrame()
destination_dataframe = original_file_dataframe

# print('hahahaha', data_file_dataframe[all_header])

destination_dataframe[all_header] = data_file_dataframe[all_header]

destination_dataframe.to_csv(exchange_folder_path + destination_file_name, sep=';')



