import os
import pandas as pd
import ezc3d
import c3d
import numpy
import math
import sys

root_path = "C:/Users/lthongkh/Documents/JNJ/"
data_path = "C:/Users/lthongkh/OneDrive - JNJ/Loco/Loco_Analysed/"
session_type = 'Crossing'
tasks = ['Complex', 'Crossing']


# session_type = 'Complex'

def replace_line(excel_file, trial):
    df_session_results = pd.read_csv(excel_file, sep=';', index_col=[0])

    df_session_results.to_csv(excel_file, sep=';')


def combine_all(session_type):
    df_all = pd.DataFrame()
    path_to_session = data_path + session_type + "_Analysed/"
    for session in os.listdir(path_to_session):
        df_session = pd.read_csv(path_to_session + session + "/Analysis/" + session + "_result.csv", sep=';')
        df_session = df_session.drop(columns=['Unnamed: 0'])
        df_all = df_all.append(df_session)
    df_all.to_csv(data_path + session_type + "_Analysed_Result/" + session_type + "_Session_result_combined_07_11_2023.csv",
                  sep=';')


def add_sessions_to_combined(sessions, session_type):
    df_all = pd.read_csv(data_path + session_type + "_Analysed_Result/" + session_type + "_Session_result_combined.csv",
                         sep=";")
    for session in sessions:
        session_csv_path = root_path + session_type + "_Analyzed/P_" + session + "_Session_" + session_type + \
                           "/Analysis/P_" + session + "_Session_" + session_type + "_result.csv"
        df_session = pd.read_csv(session_csv_path, sep=";")
        df_all = df_all.append(df_session)
    df_all = df_all.drop(columns=['Unnamed: 0'])
    df_all = df_all.sort_values(by=["session_name", "index"])
    df_all.to_csv(data_path + session_type + "_Analysed_Result/" + session_type + "_Session_result_combined.csv",
                  sep=";")

'''
def cancel_trails(filename, combined_file, task):
    df_cancel = pd.read_csv(filename, sep=',', index_col=[0])
    df_combined = pd.read_csv(combined_file, sep=";")
    df_combined = df_combined.drop(columns=['Unnamed: 0'])
    for i in range(len(df_cancel)):
        session = df_cancel.iloc[i]["session_name"]
        if len(df_cancel.iloc[i]["trial_name"].split("_")) > 1:
            trial_number = df_cancel.iloc[i]["trial_name"].split("_")[1]
        else:
            trial_number = df_cancel.iloc[i]["trial_name"]
        index = df_combined
        print(session + " " + trial_number)
        print(df_combined[(df_combined["session_name"] == session) & (df_combined["trial_name"].con trial_number)].index)
        #df_combined = df_combined.drop()'''


def combine_head_angle_calibration():
    for task in tasks:
        df_combined = pd.DataFrame()
        for session in os.listdir(data_path + task + "_Analysed/"):
            head_calib_path = data_path + task + "_Analysed/" + session + "/HeadCalibration/"
            try:
                calibration_folder = os.listdir(head_calib_path)[-1]
                data_file = head_calib_path + calibration_folder + "/Result/Result.csv"
                df_calibration = pd.read_csv(data_file, sep=";")
                df_calibration = df_calibration.drop(columns=['Unnamed: 0'])
                df_calibration['session_name'] = session.split("_")[1]
                df_combined = df_combined.append(df_calibration)
                print(df_calibration)
            except:
                print("No HeadCalibration for session " + session)
        df_combined.to_csv(root_path + "HeadCalibrationsCombined" + task + ".csv")


#cancel_trails(root_path + 'Trails_canceled.csv', data_path + "Complex_Analysed_Result/Complex_Session_result_combined_27_09.csv", "Complex")
sessions = ["2003"]
# add_sessions_to_combined(sessions, "Crossing")
#combine_head_angle_calibration()
combine_all('Crossing')


# c3d_to_csv('C:/Users/lthongkh/Documents/JNJ/Nexus-JJ/P1019/Complex/Trial_29_ComplexT_True-7_vicon_structured.c3d')


# ##### Combine all PWS calibration result
# df_PWScalibration_results = pandas.DataFrame()
# for i in range(0, len(list_full_path_sessions)):
#     path_PWScalibration_result = list_full_path_sessions[i] + '/PWSCalibration/Calibration.csv'
#     df_PWScalibration_result = pandas.read_csv(path_PWScalibration_result, sep=';', index_col=[0])
#
#     data_new_PWScalibraton = {
#         'session_name': [list_participants_name[i] + '_' + list_sessions_name[i]],
#         'subject_step_length': [df_PWScalibration_result.columns[1]],
#         'subject_feet_size': [df_PWScalibration_result.columns[2]],
#     }
#
#     df_new_PWScalibration = pandas.DataFrame(data_new_PWScalibration)
#     print(df_new_PWScalibration)
#
#     df_PWScalibration_results = df_PWScalibration_results.append(df_new_PWScalibration, ignore_index=True)
#
# calibration_result_combined_path = analyse_csv_folder_path + 'PWSCalibration_result_combined.csv'
# df_PWScalibration_results.to_csv(calibration_result_combined_path, sep=';')

# ##### Combine all body calibration result
# df_body_calibration_results = pandas.DataFrame()
# for i in range(0, len(list_full_path_sessions)):
#     all_calibration = os.listdir(list_full_path_sessions[i] + '/BodyCalibration/')
#     last_calibration = 1
#     for calibration in all_calibration:
#         if '_' in calibration:
#             last_calibration = max(int(str(calibration).split('_')[-1]), last_calibration)
#
#     path_body_calibration_result = list_full_path_sessions[i] + '/BodyCalibration/Calibration_' + str(last_calibration) + '/Vicon/Vicon_Data_In_Unity_Coordinate.csv'
#     print(path_body_calibration_result)
#     df_body_calibration_result = pandas.read_csv(path_body_calibration_result, sep=';', index_col=[0])
#
#     data_new_body_calibration = {
#         'session_name': [list_participants_name[i] + '_' + list_sessions_name[i]],
#         'subject_head_height': [df_body_calibration_result['Head_Position_y'][df_body_calibration_result['Head_Position_y'].count() - 1]],
#     }
#
#     df_new_body_calibration = pandas.DataFrame(data_new_body_calibration)
#     print(df_new_body_calibration)
#
#     df_body_calibration_results = df_body_calibration_results.append(df_new_body_calibration, ignore_index=True)
#
# calibration_result_combined_path = analyse_csv_folder_path + 'subject_height_combined.csv'
# df_body_calibration_results.to_csv(calibration_result_combined_path, sep=';')

# ##### Combine the session result with the margin_stability_variable
# df_session_results_with_MSV = pandas.DataFrame()
#
# for i in range(0, len(list_full_path_sessions)):
#     all_calibration = os.listdir(list_full_path_sessions[i] + '/ComplexCalibration/')
#     last_calibration = 1
#     for calibration in all_calibration:
#         if '_' in calibration:
#             last_calibration = max(int(str(calibration).split('_')[-1]), last_calibration)
#     path_calibration_general_result = list_full_path_sessions[i] + '/ComplexCalibration/Calibration_' + str(last_calibration) + '/Result/Calibration_' + str(last_calibration) + '_step_length_step_width.csv'
#     df_calibration_session = pandas.read_csv(path_calibration_general_result, sep=';', index_col=[0])
#     subject_step_width = df_calibration_session.at['mean', 'step_width']
#
#     # Get the list of all result folder in the session(ComplexT not Preci)
#     list_result_folder_in_session = []
#
#     path_folder_data = list_full_path_sessions[i] + '/Data/'
#     for path in os.listdir(path_folder_data):
#         if ('ComplexT' in path) and ('Preci' not in path):
#             list_result_folder_in_session.append(path_folder_data + path + '/Result/')
#
#     print(list_full_path_sessions[i])
#
#     # Get the subject_step_length and subject_feet_size for the session
#     step_foot_result = list_full_path_sessions[i] + '/ComplexCalibration/Calibration.csv'
#     df_step_foot_file = pandas.read_csv(step_foot_result, sep=';', index_col=[0])
#
#     # d = {'session_name': [str(list_participants_name[i]) + str(list_sessions_name[i])], 'subject_step_length': [list(df_step_foot_file.columns)[2]], 'subject_feet_size': [list(df_step_foot_file.columns)[3]]}
#     # df_step_foot_result = pandas.DataFrame(data=d)
#
#     # Add a file in each trial result folder contains trial result and MSV info
#     for trial_result_path in list_result_folder_in_session:
#
#         file_path_trial_result = ''
#         file_path_MSV = ''
#         file_path_maze_step_info = ''
#         file_path_trial_step_info = ''
#         trial_maze_name = ''
#
#         for file_path in os.listdir(trial_result_path):
#             if 'margin_stability_variables' in file_path:
#                 file_path_MSV = trial_result_path + file_path
#
#             if 'result' in file_path and 'with_MSV' not in file_path:
#                 file_path_trial_result = trial_result_path + file_path
#
#             if 'median' in file_path:
#                 file_path_maze_step_info = trial_result_path + file_path
#                 trial_maze_name = file_path.replace('.csv', '')
#
#             if 'step_length_step_width' in file_path:
#                 file_path_trial_step_info = trial_result_path + file_path
#
#         trial_vicon_path = trial_result_path.replace('Result', 'Vicon')
#         file_path_foot_step_maze_vicon = ''
#         for file_path in os.listdir(trial_vicon_path):
#             if 'foot_step_maze' in file_path:
#                 file_path_foot_step_maze_vicon = trial_vicon_path + file_path
#
#         # print(file_path_MSV + ' ' + file_path_trial_result)
#
#         df_file_MSV = pandas.read_csv(file_path_MSV, sep=';', index_col=[0])
#         df_file_trial_result = pandas.read_csv(file_path_trial_result, sep=';', index_col=[0])
#         df_file_maze_step_info = pandas.read_csv(file_path_maze_step_info, sep=';', index_col=[0])
#         df_file_trial_step_info = pandas.read_csv(file_path_trial_step_info, sep=';', index_col=[0])
#
#         df_file_foot_step_maze_vicon = pandas.read_csv(file_path_foot_step_maze_vicon, sep=';', index_col=[0])
#
#         # Compare the nb_step_trial and nb_step_maze
#         nb_placement_trial = len(df_file_MSV.index)
#         nb_placement_maze = len(df_file_maze_step_info.index) - 1
#
#         df_trial_result_with_MSV = pandas.DataFrame()
#
#         # Only if the nb_step_trial == nb_step_maze we save the data
#         if nb_placement_trial == nb_placement_maze:
#             # Get the general info for each session
#             df_file_trial_result_necessary = df_file_trial_result[['session_name', 'trial_name', 'task_type', 'mean_percentage_brown_pixel']]
#             df_file_trial_result_necessary['subject_step_length'] = [list(df_step_foot_file.columns)[1]]
#             df_file_trial_result_necessary['subject_step_width'] = [subject_step_width]
#             df_file_trial_result_necessary['subject_feet_size'] = [list(df_step_foot_file.columns)[2]]
#             df_file_trial_result_necessary['maze_name'] = [trial_maze_name]
#             df_file_trial_result_necessary['mean_step_length_maze'] = df_file_maze_step_info.at['mean', 'step_length']
#             df_file_trial_result_necessary['mean_step_width_maze'] = df_file_maze_step_info.at['mean', 'step_width']
#             df_file_trial_result_necessary['sd_step_length_maze'] = df_file_maze_step_info.at['standard_deviation', 'step_length']
#             df_file_trial_result_necessary['sd_step_width_maze'] = df_file_maze_step_info.at['standard_deviation', 'step_width']
#             df_file_trial_result_necessary['mean_step_length_trial'] = df_file_trial_step_info.at['mean', 'step_length']
#             df_file_trial_result_necessary['mean_step_width_trial'] = df_file_trial_step_info.at['mean', 'step_width']
#             df_file_trial_result_necessary['sd_step_length_trial'] = df_file_trial_step_info.at['standard_deviation', 'step_length']
#             df_file_trial_result_necessary['sd_step_width_trial'] = df_file_trial_step_info.at['standard_deviation', 'step_width']
#
#             df_file_trial_result_necessary = pandas.concat([df_file_trial_result_necessary] * nb_placement_trial, ignore_index=True)
#
#             # Create a dataframe with step_index
#             data = {'placement_index': range(0, nb_placement_trial)}
#             df_step_index = pandas.DataFrame(data=data)
#             # print(df_step_index)
#
#             # Get rid of the last two row of maze step and rename column and add one line at the beginning
#             df_file_maze_step_info.loc[0] = [0, 0]
#             df_file_maze_step_info = df_file_maze_step_info.drop('mean')
#             df_file_maze_step_info = df_file_maze_step_info.drop('standard_deviation')
#             df_file_maze_step_info.index = df_file_maze_step_info.index.astype(int)
#             df_file_maze_step_info = df_file_maze_step_info.sort_index()
#             df_file_maze_step_info = df_file_maze_step_info.rename(columns={"step_length": "step_length_maze", "step_width": "step_width_maze"})
#
#             # Get rid of the last two row of trial step and rename column and add one line at the beginning
#             df_file_trial_step_info.loc[0] = [0, 0]
#             df_file_trial_step_info = df_file_trial_step_info.drop('mean')
#             df_file_trial_step_info = df_file_trial_step_info.drop('standard_deviation')
#             df_file_trial_step_info.index = df_file_trial_step_info.index.astype(int)
#             df_file_trial_step_info = df_file_trial_step_info.sort_index()
#             df_file_trial_step_info = df_file_trial_step_info.rename(columns={"step_length": "step_length_trial", "step_width": "step_width_trial"})
#
#             # Attach the info of msv
#             df_file_MSV = df_file_MSV.sort_values(by=['Frame_Heel_Strike'])
#             df_file_MSV = df_file_MSV.reset_index(drop=True)
#             # df_file_MSV = df_file_MSV.drop(index=0)
#             # df_file_MSV = df_file_MSV.reset_index(drop=True)
#             df_file_MSV = df_file_MSV[['MOS_AP_Y', 'MOS_ML_X']]
#
#             # Get the step_percentage_brown_pixel for each step
#             df_file_foot_step_maze_vicon = df_file_foot_step_maze_vicon[['step_percentage_brown_pixel']]
#
#             df_trial_result_with_MSV = df_step_index.join(df_file_maze_step_info).join(df_file_trial_step_info).join(df_file_MSV).join(df_file_foot_step_maze_vicon)
#
#             df_trial_result_with_MSV = df_file_trial_result_necessary.join(df_trial_result_with_MSV, how='right')
#
#             # print(df_trial_result_with_MSV)
#
#         # for i in range(1, len(df_file_trial_result.columns)):
#         #     df_trial_result_with_MSV.insert(i - 1, df_file_trial_result.columns[i], df_file_trial_result[df_file_trial_result.columns[i]].tolist() * len(df_trial_result_with_MSV.index))
#
#         # print(df_trial_result_with_MSV)
#
#         # file_path_trial_result_with_MSV = file_path_trial_result.replace('.csv', '_with_MSV_each_step.csv')
#         # df_trial_result_with_MSV.to_csv(file_path_trial_result_with_MSV, sep=';')
#
#         # print(df_trial_result_with_MSV)
#
#         df_session_results_with_MSV = df_session_results_with_MSV.append(df_trial_result_with_MSV, ignore_index=True)
#
# print(df_session_results_with_MSV)
# df_session_results_with_MSV_path = analyse_csv_folder_path + 'Session_result_combined_with_MSV.csv'
# df_session_results_with_MSV.to_csv(df_session_results_with_MSV_path, sep=';')
#
# ##### Combine all the margin_stability_general_variables
# df_calibration_general_results = pandas.DataFrame()
#
# for i in range(0, len(list_full_path_sessions)):
#     all_calibration = os.listdir(list_full_path_sessions[i] + '/ComplexCalibration/')
#     last_calibration = 1
#     for calibration in all_calibration:
#         if '_' in calibration:
#             last_calibration = max(int(str(calibration).split('_')[-1]), last_calibration)
#     path_calibration_general_result = list_full_path_sessions[i] + '/ComplexCalibration/Calibration_' + str(last_calibration) + '/Result/Calibration_' + str(last_calibration) + '_margin_stability_general_variables.csv'
#
#     # print(path_calibration_general_result)
#
#     df_calibration_general_result = pandas.read_csv(path_calibration_general_result, sep=';', index_col=[0])
#
#     # Add the column of the session_name
#     list_session_name = [list_participants_name[i] + '_' + list_sessions_name[i]] * len(df_calibration_general_result)
#     df_calibration_general_result.insert(0, 'session_name', list_session_name)
#     # print(df_calibration_general_result)
#
#     # Add the column of PWS step length and width from heel
#     path_step_length_width_heel = list_full_path_sessions[i] + '/ComplexCalibration/Calibration_' + str(last_calibration) + '/Result/Calibration_' + str(last_calibration) + '_step_length_step_width.csv'
#     df_step_length_width_heel = pandas.read_csv(path_step_length_width_heel, sep=';', index_col=[0])
#
#     # Add the column of step_length_heel_mean
#     list_step_length_heel_mean = [df_step_length_width_heel.at['mean', 'step_length']] * len(df_calibration_general_result)
#     df_calibration_general_result.insert(1, 'step_length_heel_mean', list_step_length_heel_mean)
#
#     # Add the column of step_length_heel_sd
#     list_step_length_heel_sd = [df_step_length_width_heel.at['standard_deviation', 'step_length']] * len(df_calibration_general_result)
#     df_calibration_general_result.insert(2, 'step_length_heel_sd', list_step_length_heel_sd)
#
#     # Add the column of step_width_heel_mean
#     list_step_width_heel_mean = [df_step_length_width_heel.at['mean', 'step_width']] * len(df_calibration_general_result)
#     df_calibration_general_result.insert(3, 'step_width_heel_mean', list_step_width_heel_mean)
#
#     # Add the column of step_width_heel_sd
#     list_step_width_heel_sd = [df_step_length_width_heel.at['standard_deviation', 'step_width']] * len(df_calibration_general_result)
#     df_calibration_general_result.insert(4, 'step_width_heel_sd', list_step_width_heel_sd)
#
#     # Add the column of PWS step length from head
#     path_step_length_head = list_full_path_sessions[i] + '/ComplexCalibration/Calibration_' + str(last_calibration) + '/Result/Step_Length_Result.csv'
#     df_step_length_head = pandas.read_csv(path_step_length_head, sep=',', index_col=[0])
#
#     # Add the column of step_length_head_mean
#     list_step_width_head_mean = [df_step_length_head.at['mean', 'StepLength']] * len(df_calibration_general_result)
#     df_calibration_general_result.insert(5, 'step_length_head_mean', list_step_width_head_mean)
#
#     # Add the column of step_length_head_sd
#     step_length_head_sd = [df_step_length_head.at['std', 'StepLength']] * len(df_calibration_general_result)
#     df_calibration_general_result.insert(6, 'step_length_head_sd', step_length_head_sd)
#
#     df_calibration_general_results = df_calibration_general_results.append(df_calibration_general_result, ignore_index=True)
#
# calibration_general_result_combined_path = analyse_csv_folder_path + 'Calibration_general_result_combined.csv'
# df_calibration_general_results.to_csv(calibration_general_result_combined_path, sep=';')
#
#
#
#
#
#
#
# ##### Combine all the calibration Step_Length and Foot_Length
# df_step_foot_results = pandas.DataFrame()
# for i in range(0, len(list_full_path_sessions)):
#     step_foot_result = list_full_path_sessions[i] + '/ComplexCalibration/Calibration.csv'
#     # print(step_foot_result)
#     df_step_foot_file = pandas.read_csv(step_foot_result, sep=';', index_col=[0])
#     # print(df_step_foot_file)
#
#     d = {'session_name': [str(list_participants_name[i]) + str(list_sessions_name[i])], 'subject_step_length': [list(df_step_foot_file.columns)[2]], 'subject_feet_size': [list(df_step_foot_file.columns)[3]]}
#     df_step_foot_result = pandas.DataFrame(data=d)
#     # print(df_step_foot_result)
#
#     df_step_foot_results = df_step_foot_results.append(df_step_foot_result, ignore_index=True)
#
# # print(df_step_foot_results)
# step_foot_result_path = analyse_csv_folder_path + 'Session_step_foot_length_result.csv'
# df_step_foot_results.to_csv(step_foot_result_path, sep=';')
