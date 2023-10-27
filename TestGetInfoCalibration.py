import os
import pandas as pd
import numpy as np
from scipy.spatial import distance

#root_data_path = 'H:/JNJ/Backup/'
#result_file_path = 'H:/JNJ/Complex_Analysed/CalibrationResult.csv'
root_data_path = "C:/Users/lthongkh/OneDrive - JNJ/Loco/Loco_Analysed/Complex_Analysed/"
result_file_path = "C:/Users/lthongkh/Documents/JNJ/CalibrationResult.csv"

# root_data_path = 'E:/JNJ/Complex_Analysed/'
# result_file_path = 'E:/JNJ/Complex_Analysed/CalibrationResult.csv'

dic_calibration_result = {'session_name': [],
                          'subject_step_length': [],
                          'subject_feet_size': [],
                          'head_speed_horizontal_avg': [],
                          'subject_vicon_height': [],
                          # 'head_angle_look_straight': [],
                          # 'head_angle_look_down': [],
                          }

df_calibration_result = pd.DataFrame(dic_calibration_result)

# In case of analysing all the session in one folder
for path in os.listdir(root_data_path):
    if "Session" not in path:
        continue

    full_path = os.path.join(root_data_path, path)
    participant_name = path.rsplit('_', 1)[0]
    session_name = path.rsplit('_', 1)[1]

    # Get the subject_step_length and subject_feet_size from the Calibration.csv file in PWSCalibration
    pws_calibration_result_path = full_path + '/PWSCalibration/Calibration.csv'
    df_pws_calibration_result = pd.read_csv(pws_calibration_result_path, sep=';')
    subject_step_length = df_pws_calibration_result.columns[2]
    subject_feet_size = df_pws_calibration_result.columns[3]

    # Get the head_speed_horizontal_avg from the last PWSCalibration
    all_pws_calibrations = os.listdir(full_path + '/PWSCalibration/')
    last_pws_calibration = 1
    for calibration in all_pws_calibrations:
        if '_' in calibration:
            last_pws_calibration = max(int(str(calibration).split('_')[-1]), last_pws_calibration)

    path_pws_vicon_data = full_path + '/PWSCalibration/Calibration_' + str(
        last_pws_calibration) + '/Vicon/Vicon_Data_In_Unity_Coordinate.csv'
    # print(path_pws_vicon_data)

    ## The +z value for the walking zone threshold (used to truncate the data)
    walking_zone_threshold_positive_z = 1

    ## The -z value for the walking zone threshold (used to truncate the data)
    walking_zone_threshold_negative_z = -2

    ## The path of tha vicon data file
    vicon_data = pd.read_csv(path_pws_vicon_data, delimiter=';', dtype='str')

    ## replace , by .
    vicon_data.replace(to_replace=',', value='.', regex=True, inplace=True)

    ## Convert data type
    vicon_data['Head_Position_x'] = pd.to_numeric(vicon_data['Head_Position_x'], errors='coerce')
    vicon_data['Head_Position_z'] = pd.to_numeric(vicon_data['Head_Position_z'], errors='coerce')

    ## Pick the list of data points that the head is in the walking zone (index)
    head_in_threshold = [i for i in range(vicon_data.shape[0]) if
                         walking_zone_threshold_negative_z <= vicon_data['Head_Position_z'][
                             i] <= walking_zone_threshold_positive_z]

    diff_abs_head_x = abs(vicon_data['Head_Position_x'].diff().fillna(0.))
    diff_abs_head_x_in_threshold = diff_abs_head_x[head_in_threshold]

    diff_abs_head_z = abs(vicon_data['Head_Position_z'].diff().fillna(0.))
    diff_abs_head_z_in_threshold = diff_abs_head_z[head_in_threshold]

    diff_xz_plane_in_threshold = np.sqrt(diff_abs_head_x_in_threshold ** 2 + diff_abs_head_z_in_threshold ** 2)

    head_speed_horizontal_avg = np.array(diff_xz_plane_in_threshold).mean() / (1 / 120.)

    # Get the subject_vicon_height from the body calibration
    all_body_calibrations = os.listdir(full_path + '/BodyCalibration/')
    last_body_calibration = 1
    for calibration in all_body_calibrations:
        if '_' in calibration:
            last_body_calibration = max(int(str(calibration).split('_')[-1]), last_body_calibration)

    path_body_vicon_data = full_path + '/BodyCalibration/Calibration_' + str(
        last_body_calibration) + '/Vicon/Vicon_Data_In_Vicon_Coordinate.csv'

    try:
        df_body_vicon_data = pd.read_csv(path_body_vicon_data, sep=';')
        df_body_vicon_data['DateTime'] = pd.to_datetime(df_body_vicon_data['DateTime'], errors='coerce')
    except:
        last_body_calibration -= 1
        path_body_vicon_data = full_path + '/BodyCalibration/Calibration_' + str(
            last_body_calibration) + '/Vicon/Vicon_Data_In_Vicon_Coordinate.csv'
        df_body_vicon_data = pd.read_csv(path_body_vicon_data, sep=';')
        df_body_vicon_data['DateTime'] = pd.to_datetime(df_body_vicon_data['DateTime'], errors='coerce')

    # Get the datetime for the event LeftTragusCalibration
    path_body_event_data = full_path + '/BodyCalibration/Calibration_' + str(
        last_body_calibration) + '/Event/Event.csv'

    df_body_event_data = pd.read_csv(path_body_event_data, sep=';')
    df_body_event_data['DateTime'] = pd.to_datetime(df_body_event_data['DateTime'], errors='coerce')

    df_left_tragus = df_body_event_data[df_body_event_data['EventName'] == 'LeftTragusCalibration']
    df_left_tragus = df_left_tragus[df_body_event_data['EventType'] == 'Start']

    datetime_left_tragus = df_left_tragus['DateTime']

    # print('datetime_left_tragus', datetime_left_tragus)

    # Get the index of the left tragus event
    datetime_list_body = pd.Index(sorted(set(df_body_vicon_data['DateTime'])))
    # print('datetime_list_body', datetime_list_body)
    index_left_tragus = datetime_list_body.get_indexer(datetime_left_tragus, limit=1, method='nearest')
    index_left_tragus = index_left_tragus[0]
    # print('index_left_tragus', index_left_tragus)

    try:
        subject_vicon_height = df_body_vicon_data['Head_Position_z'][index_left_tragus]
    except:
        subject_vicon_height = 0
        print("No HEIGHT")



    # # Get the head_angle_look_straight and the head_angle_look_down from the head calibration
    # all_head_calibrations = os.listdir(full_path + '/HeadCalibration/')
    # last_head_calibration = 1
    # for calibration in all_head_calibrations:
    #     if '_' in calibration:
    #         last_head_calibration = max(int(str(calibration).split('_')[-1]), last_head_calibration)
    #
    # path_head_result_data = full_path + '/HeadCalibration/Calibration_' + str(
    #     last_head_calibration) + '/Result/Result.csv'
    #
    # df_head_result_data = pd.read_csv(path_head_result_data, sep=';')
    # head_angle_look_straight = df_head_result_data['head_angle_look_straight'][0]
    # head_angle_look_down = df_head_result_data['head_angle_look_down'][0]

    # Gathering all the data
    dic_session_calibration_data = {'session_name': [participant_name + '_' + session_name],
                                    'subject_step_length': [subject_step_length],
                                    'subject_feet_size': [subject_feet_size],
                                    'head_speed_horizontal_avg': [head_speed_horizontal_avg],
                                    'subject_vicon_height': [subject_vicon_height],
                                    # 'head_angle_look_straight': [head_angle_look_straight],
                                    # 'head_angle_look_down': [head_angle_look_down],
                                    }

    print(dic_session_calibration_data)
    df_session_calibration_data = pd.DataFrame(dic_session_calibration_data)

    df_calibration_result = df_calibration_result.append(df_session_calibration_data, ignore_index=True)

df_calibration_result.to_csv(result_file_path, sep=';')

# 117