import collections.abc
import os
import shutil

import c3d
import pandas as pd
import re

'''
Process:
- Run replace_old_csv_data()
- Rerun Complex and Crossing analysis
- Rename_combined()
- Copy local files to OneDrive
- Rename_combined() back
- Run loop with replace_combine_one_line() (change file name to date)
'''

nexus_folder = 'C:/Users/lthongkh/OneDrive - JNJ/Loco/Nexus-JJ'
analysis_path = 'C:/Users/lthongkh/Documents/JNJ/clinical_study_'
data_path = 'C:/Users/lthongkh/OneDrive - JNJ/Loco/Loco_Analysed/'


def copy_csv(src, dst):
    df = pd.read_csv(src, sep=';')
    df.to_csv(dst, sep=';', index=False)


def is_point_null(point_all_frame, index_frame, index):
    return point_all_frame[index_frame][index][0] == 0 and point_all_frame[index_frame][index][1] == 0 and point_all_frame[index_frame][index][2] == 0

def replace_old_csv_data():
    for session_folder in os.listdir(nexus_folder):
        print("Session folder", session_folder)
        if session_folder != ".Done":
            for task_folder in os.listdir(nexus_folder + '/' + session_folder):
                task_path = nexus_folder + '/' + session_folder + '/' + task_folder
                for file in os.listdir(task_path):
                    if file[-3:] == 'c3d':

                        trial_id = file.split('_')[1]
                        csv_file_path = (task_path + '/' + file).replace('.c3d', '.csv')
                        print("Trial index", trial_id)
                        print(csv_file_path)

                        dictionary_vicon_coordinate = {}

                        reader = c3d.Reader(open(task_path + '/' + file, 'rb'))
                        point_labels = reader.point_labels

                        points_all_frame = []
                        for frame_no, points, analog in reader.read_frames():
                            points_all_frame.append(points)

                        i = 0
                        for point_label in point_labels:
                            if ('Marker' not in point_label):
                                # _Occluded
                                list_occluded = []
                                for index_frame in range(len(points_all_frame)):
                                    list_occluded.append('False')

                                dictionary_vicon_coordinate[
                                    (point_label + '_Occluded').replace(' ', '')] = list_occluded


                            closest_non_null = 0
                            while is_point_null(points_all_frame, closest_non_null, i):
                                closest_non_null += 1

                            # _Position_x_y_z
                            list_x_positions = []
                            list_y_positions = []
                            list_z_positions = []
                            for index_frame in range(len(points_all_frame)):
                                if not is_point_null(points_all_frame, index_frame, i):
                                    list_x_positions.append(points_all_frame[index_frame][i][0] / 1000)
                                    list_y_positions.append(points_all_frame[index_frame][i][1] / 1000)
                                    list_z_positions.append(points_all_frame[index_frame][i][2] / 1000)
                                    closest_non_null = index_frame
                                else:
                                    list_x_positions.append(points_all_frame[closest_non_null][i][0] / 1000)
                                    list_y_positions.append(points_all_frame[closest_non_null][i][1] / 1000)
                                    list_z_positions.append(points_all_frame[closest_non_null][i][2] / 1000)

                            dictionary_vicon_coordinate[
                                (point_label + '_Position_x').replace(' ', '')] = list_x_positions
                            dictionary_vicon_coordinate[
                                (point_label + '_Position_y').replace(' ', '')] = list_y_positions
                            dictionary_vicon_coordinate[
                                (point_label + '_Position_z').replace(' ', '')] = list_z_positions

                            if ('Marker' not in point_label):

                                # _Quaternion_x, _Quaternion_y, _Quaternion_z, _Quaternion_w
                                list_quaternion_x = []
                                list_quaternion_y = []
                                list_quaternion_z = []
                                list_quaternion_w = []

                                for index_frame in range(len(points_all_frame)):
                                    list_quaternion_x.append(0)
                                    list_quaternion_y.append(0)
                                    list_quaternion_z.append(1)
                                    list_quaternion_w.append(0)

                                dictionary_vicon_coordinate[
                                    (point_label + '_Quaternion_x').replace(' ', '')] = list_quaternion_x
                                dictionary_vicon_coordinate[
                                    (point_label + '_Quaternion_y').replace(' ', '')] = list_quaternion_y
                                dictionary_vicon_coordinate[
                                    (point_label + '_Quaternion_z').replace(' ', '')] = list_quaternion_z
                                dictionary_vicon_coordinate[
                                    (point_label + '_Quaternion_w').replace(' ', '')] = list_quaternion_w

                            i = i + 1

                        vicon_data_path = data_path + task_folder + '_Analysed/P_' + session_folder + '_Session_' + task_folder + '/Data/'
                        trial_folder = ''

                        for trial_folder_possible in os.listdir(vicon_data_path):
                            if trial_folder_possible.split('_')[1] == trial_id:
                                trial_folder = trial_folder_possible

                        # Replace Vicon csv file in data
                        vicon_data_path = data_path + task_folder + '_Analysed/P_' + session_folder + '_Session_' + task_folder + '/Data/' + trial_folder + '/Vicon/Vicon_Data_In_Vicon_Coordinate.csv'
                        try:
                            os.rename(vicon_data_path, vicon_data_path[:-4] + '_OLD.csv')
                        except:
                            print("file", vicon_data_path, vicon_data_path[:-4] + '_OLD.csv', "already exists")
                        df_old_vicon_coordinate = pd.read_csv(vicon_data_path[:-4] + '_OLD.csv', sep=';')
                        dictionary_vicon_coordinate['DateTime'] = df_old_vicon_coordinate['DateTime']
                        dictionary_vicon_coordinate['TimeElapsed'] = df_old_vicon_coordinate['TimeElapsed']
                        dictionary_vicon_coordinate['Latency'] = df_old_vicon_coordinate['Latency']
                        dictionary_vicon_coordinate['Timecode'] = df_old_vicon_coordinate['Timecode']
                        dictionary_vicon_coordinate['Framerate'] = df_old_vicon_coordinate['Framerate']
                        dictionary_vicon_coordinate['HeadCalibrated_Quaternion_x'] = df_old_vicon_coordinate[
                            'HeadCalibrated_Quaternion_x']
                        dictionary_vicon_coordinate['HeadCalibrated_Quaternion_y'] = df_old_vicon_coordinate[
                            'HeadCalibrated_Quaternion_y']
                        dictionary_vicon_coordinate['HeadCalibrated_Quaternion_z'] = df_old_vicon_coordinate[
                            'HeadCalibrated_Quaternion_z']
                        dictionary_vicon_coordinate['HeadCalibrated_Quaternion_w'] = df_old_vicon_coordinate[
                            'HeadCalibrated_Quaternion_w']
                        df_new_vicon_coordinate = pd.DataFrame(data=dictionary_vicon_coordinate)
                        df_new_vicon_coordinate.to_csv(vicon_data_path, sep=';')

                        # Copy files to location to be analyzed again
                        analysis_folder = analysis_path + task_folder + "/P_" + session_folder + '_Session_' + task_folder
                        src = data_path + task_folder + '_Analysed/P_' + session_folder + '_Session_' + task_folder
                        if not os.path.exists(analysis_folder):
                            os.mkdir(analysis_folder)
                            os.mkdir(analysis_folder + "/Data/")
                            os.mkdir(analysis_folder + "/Archived")
                            os.mkdir(analysis_folder + "/Configuration")
                            os.mkdir(analysis_folder + "/BodyCalibration")
                            if task_folder == "Complex":
                                os.mkdir(analysis_folder + "/PWSCalibration")
                                os.mkdir(analysis_folder + "/HeadCalibration")

                        try:
                            os.mkdir(analysis_folder + "/Data/" + trial_folder)
                            os.mkdir(analysis_folder + "/Data/" + trial_folder + '/Result')
                            os.mkdir(analysis_folder + "/Data/" + trial_folder + '/Vicon')
                        except:
                            print("Folder Data already exists")

                        copy_csv(
                            src + "/Configuration/S" + session_folder + "_Config_" + task_folder + "_Clinical_Trial.csv",
                            analysis_folder + "/Configuration/S" + session_folder + "_Config_" + task_folder + "_Clinical_Trial.csv")
                        if task_folder == "Complex":
                            copy_csv(src + "/PWSCalibration/Calibration.csv",
                                     analysis_folder + "/PWSCalibration/Calibration.csv")
                            for file_to_copy in os.listdir(src + "/HeadCalibration/"):
                                try:
                                    os.mkdir(analysis_folder + "/HeadCalibration/" + file_to_copy)
                                    os.mkdir(analysis_folder + "/HeadCalibration/" + file_to_copy + "/Vicon/")
                                    copy_csv(
                                        src + "/HeadCalibration/" + file_to_copy + "/Vicon/Vicon_Data_In_Vicon_Coordinate.csv",
                                        analysis_folder + "/HeadCalibration/" + file_to_copy + "/Vicon/Vicon_Data_In_Vicon_Coordinate.csv")
                                except:
                                    print("File",
                                          analysis_folder + "/HeadCalibration/" + file_to_copy + "/Vicon/Vicon_Data_In_Vicon_Coordinate.csv",
                                          "already exists")
                        if task_folder == "Crossing":
                            copy_csv(src + "/Data/" + trial_folder + "/Result/Result.csv",
                                     analysis_folder + "/Data/" + trial_folder + "/Result/Result.csv")
                        for file_to_copy in os.listdir(src + "/Data/" + trial_folder + "/Vicon/"):
                            if file_to_copy[:5] == 'Vicon':
                                print(file_to_copy)
                                copy_csv(src + "/Data/" + trial_folder + "/Vicon/" + file_to_copy,
                                         analysis_folder + "/Data/" + trial_folder + "/Vicon/" + file_to_copy)
                        for file_to_copy in os.listdir(src + "/BodyCalibration/"):
                            try:
                                os.mkdir(analysis_folder + "/BodyCalibration/" + file_to_copy)
                            except:
                                print(analysis_folder + "/BodyCalibration/" + file_to_copy, "already exists")
                            try:
                                os.mkdir(analysis_folder + "/BodyCalibration/" + file_to_copy + "/Vicon/")
                            except:
                                print(analysis_folder + "/BodyCalibration/" + file_to_copy + "/Vicon/", "already exists")
                            print(file_to_copy)
                            copy_csv(src + "/BodyCalibration/" + file_to_copy + "/Vicon/Vicon_Data_In_Vicon_Coordinate.csv",
                                     analysis_folder + "/BodyCalibration/" + file_to_copy + "/Vicon/Vicon_Data_In_Vicon_Coordinate.csv")


'''Method called to replace specific trials data in
the combined file'''
def replace_combine_one_line(session, task, old_date, new_date):
    new_df = pd.read_csv(
        analysis_path + task + '/P_' + session + '_Session_' + task + '/Analysis/P_' + session + '_Session_' + task + '_result.csv',
        sep=';')
    new_df = new_df.drop(columns=['Unnamed: 0'])
    current_df = pd.read_csv(
        data_path + task + '_Analysed/P_' + session + '_Session_' + task + '/Analysis/P_' + session + '_Session_' + task + '_result.csv',
        sep=';')
    current_df = current_df.drop(columns=['Unnamed: 0'])
    old_df = pd.read_csv(data_path + task + "_Analysed_Result/" + task + "_Session_result_combined_" + old_date + ".csv", sep=';')
    old_df = old_df.drop(columns=['Unnamed: 0'])
    session_name = new_df["session_name"][0]

    for trial_to_transfer in new_df["trial_name"]:
        index = current_df[current_df["trial_name"] == trial_to_transfer].index
        index_all = old_df[(old_df["trial_name"] == trial_to_transfer) & (old_df["session_name"] == session_name)].index
        if task == 'Crossing/':
            new_df[new_df['trial_name'] == trial_to_transfer]["crossing_1_Reasons_Archived_Trial"] = current_df[current_df["trial_name"] == trial_to_transfer]["crossing_1_Reasons_Archived_Trial"].values[0]
        for column in current_df.columns:
            current_df.at[index.values[0], column] = new_df[new_df["trial_name"] == trial_to_transfer][column].values[0]
        for column in old_df.columns:
            if len(index_all.values) > 0:
                old_df.at[index_all.values[0], column] = \
                    new_df[new_df["trial_name"] == trial_to_transfer][column].values[0]

    try:
        os.rename(
            data_path + task + '_Analysed/P_' + session + '_Session_' + task + '/Analysis/P_' + session + '_Session_' + task + '_result.csv',
            data_path + task + '_Analysed/P_' + session + '_Session_' + task + '/Analysis/P_' + session + '_Session_' + task + '_result_OLD.csv')
    except:
        print("file",
              data_path + task + '_Analysed/P_' + session + '_Session_' + task + '/Analysis/P_' + session + '_Session_' + task + '_result_OLD.csv',
              "already exists")
    current_df.to_csv(
        data_path + task + '_Analysed/P_' + session + '_Session_' + task + '/Analysis/P_' + session + '_Session_' + task + '_result.csv',
        sep=';')
    old_df.to_csv(data_path + task + "_Analysed_Result/" + task + "_Session_result_combined_" + new_date + ".csv", sep=';')


# To use in case files were downloaded to OneDrive before
# running the replace_combine_one_line() loop
def correct(list):
    path = 'C:/Users/lthongkh/OneDrive - JNJ/Loco/Loco_Analysed/Complex_Analysed/P_'
    mid_path = "_Session_Complex/Analysis/P_"
    for session in list:
        print("Session", session)
        csv_path = path + str(session) + mid_path + str(session) + "_Session_Complex_result.csv"
        csv_old_path = path + str(session) + mid_path + str(session) + "_Session_Complex_result_OLD.csv"

        new_df = pd.read_csv(csv_path, sep=';')
        new_df = new_df.drop(columns=['Unnamed: 0'])
        old_df = pd.read_csv(csv_old_path, sep=';')
        old_df = old_df.drop(columns=['Unnamed: 0'])

        for trial_to_transfer in new_df["trial_name"]:
            index = old_df[old_df["trial_name"] == trial_to_transfer].index
            for column in old_df.columns:
                old_df.at[index.values[0], column] = new_df[new_df["trial_name"] == trial_to_transfer][column].values[0]

        old_df.to_csv(csv_path, sep=';')


def rename_combined():
    tasks = ["Complex/", "Crossing/"]
    for task in tasks:
        for session_folder in os.listdir(analysis_path + task):
            os.rename(analysis_path + task + session_folder + "/Analysis/" + session_folder + "_result_NEW.csv",
                      analysis_path + task + session_folder + "/Analysis/" + session_folder + "_result.csv")
    return


#replace_old_csv_data()
#rename_combined()
for session_folder in os.listdir(analysis_path + "Complex/"):
    replace_combine_one_line(session_folder.split("_")[1], "Complex", "15_11_2023", "16_11_2023")
