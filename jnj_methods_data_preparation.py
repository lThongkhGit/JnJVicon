import pandas as pd
import numpy as np
import jnj_methods_analysis as analysis
import glob
import csv
import os
import itertools
import c3d
import scipy
import ezc3d

# The name of the left foot toe marker
name_left_toe_marker = 'LeftToeCalibrated'

# The name of the left foot heel marker
name_left_heel_marker = 'LeftHeelCalibrated'

# The name of the right foot toe marker
name_right_toe_marker = 'RightToeCalibrated'

# The name of the right foot heel marker
name_right_heel_marker = 'RightHeelCalibrated'

# The name of the LASI Pelvis Front marker
name_LASI_marker = 'LASICalibrated'

# The name of the RASI Pelvis Front marker
name_RASI_marker = 'RASICalibrated'

# The name of the LPSI Pelvis Back marker
name_LPSI_marker = 'LPSICalibrated'

# The name of the RPSI Pelvis Back marker
name_RPSI_marker = 'RPSICalibrated'

# The name of the RPSI Pelvis Back marker
name_pelvis_marker = 'PelvisCalibrated'

# Frame per second
fps = 120.

##### Create MultiIndex for Vicon data
# MultiIndex the vicon dataframe by
# DateTime
# TimeElapsed
# Latency
# Timecode
# Framerate
# Rigidbody Name
#   -Occluded
#   -PosX
#   -PosY
#   -PosZ
#   -QuaX
#   -QuaY
#   -QuaZ
#   -QuaW
# Marker Name
#   -PosX
#   -PosY
#   -PosZ

# The list of column name representing metadata in the dataframe
vicon_columns_meta = ['DateTime', 'TimeElapsed', 'Latency', 'Timecode', 'Framerate']

# The list of column name representing position in the dataframe
position_iterables = ['PosX', 'PosY', 'PosZ']

# The list of column name representing quaternion in the dataframe
quaternion_iterables = ['QuaX', 'QuaY', 'QuaZ', 'QuaW']

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


class SessionData:
    def __init__(self, root_data_path_, participant_id_, session_id_):
        self.participant_id = participant_id_
        self.session_id = session_id_
        # The name of the session, for example, P_001_Session1
        self.name = self.participant_id + '_' + self.session_id
        # The path of the session folder
        self.session_folder_path = root_data_path_ + self.participant_id + '_' + self.session_id + '/'

        # The path of the analysis debug plot folder
        self.session_plot_folder_path = self.session_folder_path + 'Analysis/Plot/'
        if not os.path.exists(self.session_plot_folder_path):
            os.makedirs(self.session_plot_folder_path)

        # The path of the session result data
        self.session_result_data_path = self.session_folder_path + 'Analysis/' + self.name + '_result.csv'

        # The path of the config file for this session
        self.config_file_path = glob.glob(self.session_folder_path + 'Configuration' + '/*.csv')[0]

        # The result dataframe for the session
        self.df_session_result = None

        # Load Session config file
        try:
            self.df_config = pd.read_csv(self.config_file_path, sep=';')
        except FileNotFoundError:
            print('The session folder for ' + self.participant_id + '_' + self.session_id + ' does not exist!')

        # The creation of trial data list is handled in child class
        self.list_trials = []

    # Export results of all trials to csv file
    def export_session_result(self):
        # Load results of all trials
        list_all_trials_result = [trial_.result.df_trial_result for trial_ in self.list_trials]
        # Return if all objects in the list_all_trials_result are none
        if all(result is None for result in list_all_trials_result):
            return
        # perform dataframe concatenation operations
        self.df_session_result = pd.concat(list_all_trials_result)
        # Reindex the dataframe
        self.df_session_result.index = np.arange(1, len(self.df_session_result) + 1)
        # Save session result dataframe to csv file
        self.df_session_result.to_csv(self.session_result_data_path, sep=';')


class TrialData:
    def __init__(self, session_):
        # The SessionData instance
        self.session = session_

        # The list of rigidbody that are present in the trial
        self.list_rigidbody_in_trial = []

        # The list of markers that are present in the trial
        self.list_marker_in_trial = []

        # The MultiIndex to restructure the vicon dataframe
        self.rigidbody_multi_index = None
        self.marker_multi_index = None

        # The raw vicon dataframe
        self.df_vicon_raw = None

        # The structured vicon dataframe
        self.df_vicon_structured = None

        # The filtered vicon dataframe
        self.df_vicon_filtered = None

        # The TrialConfig instance is handled in child class
        self.config = None

        # The path of the trial folder is handled in child class
        self.trial_folder_path = None

        # The result instance is handled in the child class
        self.result = None

        # the number of peak detected and deleted for the main rigidbody
        self.num_peak_deleted = 0

    # data preparation:
    # read vicon data, restructure and filter data
    def data_preparation(self):
        self.prepare_vicon()
        self.filter_vicon_position()

    # Load vicon trial data, then generate csv and pickle files
    def prepare_vicon(self):
        # Load Trial data
        try:
            self.df_vicon_raw = pd.read_csv(self.trial_folder_path + 'Vicon/Vicon_Data_In_Vicon_Coordinate.csv', sep=';')

            # Delete the column 5 Marker Wand & L-FrameOLD_Occluded
            self.df_vicon_raw = self.df_vicon_raw.drop(
                ['5 Marker Wand & L-FrameOLD_Occluded', '5 Marker Wand & L-FrameOLD_Position_x', '5 Marker Wand & L-FrameOLD_Position_y', '5 Marker Wand & L-FrameOLD_Position_z',
                 '5 Marker Wand & L-FrameOLD_Quaternion_x', '5 Marker Wand & L-FrameOLD_Quaternion_y', '5 Marker Wand & L-FrameOLD_Quaternion_z', '5 Marker Wand & L-FrameOLD_Quaternion_w',
                 '5 Marker Wand & L-FrameOLD_Marker_MarkerA_Position_x', '5 Marker Wand & L-FrameOLD_Marker_MarkerA_Position_y', '5 Marker Wand & L-FrameOLD_Marker_MarkerA_Position_z',
                 '5 Marker Wand & L-FrameOLD_Marker_MarkerB_Position_x', '5 Marker Wand & L-FrameOLD_Marker_MarkerB_Position_y', '5 Marker Wand & L-FrameOLD_Marker_MarkerB_Position_z',
                 '5 Marker Wand & L-FrameOLD_Marker_MarkerC_Position_x', '5 Marker Wand & L-FrameOLD_Marker_MarkerC_Position_y', '5 Marker Wand & L-FrameOLD_Marker_MarkerC_Position_z',
                 '5 Marker Wand & L-FrameOLD_Marker_MarkerD_Position_x', '5 Marker Wand & L-FrameOLD_Marker_MarkerD_Position_y', '5 Marker Wand & L-FrameOLD_Marker_MarkerD_Position_z',
                 '5 Marker Wand & L-FrameOLD_Marker_MarkerE_Position_x', '5 Marker Wand & L-FrameOLD_Marker_MarkerE_Position_y', '5 Marker Wand & L-FrameOLD_Marker_MarkerE_Position_z'], axis=1,
                errors='ignore')

            print('Preparing Vicon data for ' + self.config.name + ' in session ' + self.session.name)
        except FileNotFoundError:
            print('The trial folder for ' + self.config.name + ' in session ' + self.session.name + ' does not exist!')
            return
        except pd.errors.EmptyDataError:
            print('The file Vicon_Data_In_Vicon_Coordinate.csv for ' + self.config.name + ' in session ' + self.session.name + ' is empty!')
            return

        '''
        Data preparation for vicon raw dataframe
        '''

        # # Remove duplicated columns in the vicon raw dataframe
        # self.df_vicon_raw = self.df_vicon_raw.loc[:, ~self.df_vicon_raw.columns.str.replace("(\.\d+)$", "").duplicated()]

        # Replace the ',' by '.' in the float
        self.df_vicon_raw.replace(to_replace=',', value='.', regex=True, inplace=True)

        # Get column information for list_rigidbody_in_trial, list_marker_in_trial, rigidbody_multi_index and marker_multi_index
        self.extract_column_information_vicon()

        '''
        Construct vicon structured dataframe with multi index
        '''
        # Calculate the number of columns in the multi index
        n_columns_multi_index = (self.rigidbody_multi_index.levshape[0] * self.rigidbody_multi_index.levshape[1]) + (self.marker_multi_index.levshape[0] * self.marker_multi_index.levshape[1])

        # Append the multi index with all indices of rigidbody and marker
        all_multi_index = self.rigidbody_multi_index.append(self.marker_multi_index)

        # Create the empty vicon structured dataframe
        self.df_vicon_structured = pd.DataFrame(np.zeros((len(self.df_vicon_raw.index), n_columns_multi_index)),
                                                columns=all_multi_index)

        # Feed the vicon structured dataframe with meta columns
        for meta in vicon_columns_meta:
            self.df_vicon_structured[meta] = self.df_vicon_raw[meta]

        # Feed the rigidbody columns
        for rigidbody in self.list_rigidbody_in_trial:
            # Convert position and quaternion columns to float (If ‘coerce’, then invalid parsing will be set as NaN.)
            for element in ['_Position_x', '_Position_y', '_Position_z', '_Quaternion_x', '_Quaternion_y',
                            '_Quaternion_z', '_Quaternion_w']:
                self.df_vicon_raw[rigidbody + element] = pd.to_numeric(self.df_vicon_raw[rigidbody + element],
                                                                       errors='coerce')

            # Convert meter to mm
            for element in ['_Position_x', '_Position_y', '_Position_z']:
                self.df_vicon_raw[rigidbody + element] = self.df_vicon_raw[rigidbody + element].apply(
                    lambda x: x * 1000)

            # print(rigidbody)

            self.df_vicon_structured.loc[:, (rigidbody, 'Occluded')] = self.df_vicon_raw[rigidbody + '_Occluded']
            self.df_vicon_structured.loc[:, (rigidbody, 'PosX')] = self.df_vicon_raw[rigidbody + '_Position_x']
            self.df_vicon_structured.loc[:, (rigidbody, 'PosY')] = self.df_vicon_raw[rigidbody + '_Position_y']
            self.df_vicon_structured.loc[:, (rigidbody, 'PosZ')] = self.df_vicon_raw[rigidbody + '_Position_z']
            self.df_vicon_structured.loc[:, (rigidbody, 'QuaX')] = self.df_vicon_raw[rigidbody + '_Quaternion_x']
            self.df_vicon_structured.loc[:, (rigidbody, 'QuaY')] = self.df_vicon_raw[rigidbody + '_Quaternion_y']
            self.df_vicon_structured.loc[:, (rigidbody, 'QuaZ')] = self.df_vicon_raw[rigidbody + '_Quaternion_z']
            self.df_vicon_structured.loc[:, (rigidbody, 'QuaW')] = self.df_vicon_raw[rigidbody + '_Quaternion_w']

        # Feed the presenting marker columns
        for marker in self.list_marker_in_trial:
            for element in ['_Position_x', '_Position_y', '_Position_z']:
                # Convert position columns to float (If ‘coerce’, then invalid parsing will be set as NaN.)
                self.df_vicon_raw[marker + element] = pd.to_numeric(self.df_vicon_raw[marker + element],
                                                                    errors='coerce')

                # Convert meter to mm
                self.df_vicon_raw[marker + element] = self.df_vicon_raw[marker + element].apply(lambda x: x * 1000)

            self.df_vicon_structured.loc[:, (marker, 'PosX')] = self.df_vicon_raw[marker + '_Position_x']
            self.df_vicon_structured.loc[:, (marker, 'PosY')] = self.df_vicon_raw[marker + '_Position_y']
            self.df_vicon_structured.loc[:, (marker, 'PosZ')] = self.df_vicon_raw[marker + '_Position_z']

        # Convert TimeElapsed column to float (If ‘coerce’, then invalid parsing will be set as NaN.)
        self.df_vicon_structured['TimeElapsed'] = pd.to_numeric(self.df_vicon_structured['TimeElapsed'], errors='coerce')

        # Save raw vicon dataframe to csv file
        self.df_vicon_raw.to_csv(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_raw.csv', sep=';')

        # Save raw vicon dataframe to pickle file
        self.df_vicon_raw.to_pickle(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_raw.pkl')

        # Save structured vicon dataframe to csv file
        self.df_vicon_structured.to_csv(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_structured.csv', sep=';')

        # Save structured vicon dataframe to pickle file
        self.df_vicon_structured.to_pickle(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_structured.pkl')

    # Get information of list_rigidbody_in_trial, list_marker_in_trial, rigidbody_multi_index and marker_multi_index
    # from vicon raw data
    def extract_column_information_vicon(self):
        # If the df_vicon_raw is None, try to load .pkl file
        if not self.load_vicon_raw():
            return

        # The list of names representing the rigidbodies in the trial
        self.list_rigidbody_in_trial = list(
            filter(lambda c: 'Occluded' in c and np.count_nonzero(self.df_vicon_raw[c]) < len(self.df_vicon_raw.index),
                   self.df_vicon_raw.columns))
        self.list_rigidbody_in_trial = [s.replace('_Occluded', '') for s in self.list_rigidbody_in_trial]

        # The list of names representing the markers in the trial
        self.list_marker_in_trial = list(filter(
            lambda c: 'Marker' in c and 'Position_x' in c and np.count_nonzero(self.df_vicon_raw[c] == 'NA') < len(
                self.df_vicon_raw.index), self.df_vicon_raw.columns))

        self.list_marker_in_trial = [s.replace('_Position_x', '') for s in self.list_marker_in_trial]

        # Restructure the vicon dataframe for MultiIndex
        self.rigidbody_multi_index = pd.MultiIndex.from_product([self.list_rigidbody_in_trial, list(
            itertools.chain(['Occluded'], position_iterables, quaternion_iterables))], names=['first', 'second'])
        self.marker_multi_index = pd.MultiIndex.from_product([self.list_marker_in_trial, position_iterables],
                                                             names=['first', 'second'])

    # Filter position of the markers and rigidbodies with spline and butterworth filter in vicon data
    def filter_vicon_position(self):
        # If the df_vicon_structured is None, try to load .pkl file
        if not self.load_vicon_structured():
            return False

        # Make a deep copy of the df_vicon_structured
        self.df_vicon_filtered = self.df_vicon_structured.copy()

        # Filter data in marker list
        for marker in self.list_marker_in_trial:
            # Convert the position to speed. Find the peaks and delete them, height = 0.1
            self.df_vicon_filtered.loc[:, (marker, position_iterables)] = analysis.detect_peak_delete(
                self, self.df_vicon_filtered.loc[:, (marker, position_iterables)].to_numpy(), freq=100)

            # Fill the gap with spline
            self.df_vicon_filtered.loc[:, (marker, position_iterables)] = analysis.fill_gap_trash(
                self.df_vicon_filtered.loc[:, (marker, position_iterables)].to_numpy(), kind='linear')

            # Filter the data with butterworth filter
            for ax in position_iterables:
                self.df_vicon_filtered.loc[:, (marker, ax)] = analysis.filt_my_signal(
                    self.df_vicon_filtered.loc[:, (marker, ax)], freq=100, lp=7, n=4, plotit=False)

        # Filter data in rigidbody list
        for rigidbody in self.list_rigidbody_in_trial:
            # Configuration for the plot
            rigidbody_list_plot = ['PelvisCalibrated', 'LeftToeCalibrated', 'RightToeCalibrated', 'LeftHeelCalibrated', 'RightHeelCalibrated']
            ax_plot = 'PosZ'

            if rigidbody in rigidbody_list_plot:
                plot_or_no = True
            else:
                plot_or_no = False

            # Convert the position to speed. Find the peaks and delete them, height = 0.1
            plot_name = rigidbody + '_detect_peak_delete_' + ax_plot
            plot_path = self.trial_folder_path + 'Vicon/' + plot_name + '.png'
            self.df_vicon_filtered.loc[:, (rigidbody, position_iterables)] = analysis.detect_peak_delete(
                self, self.df_vicon_filtered.loc[:, (rigidbody, position_iterables)].to_numpy(), freq=120, plotit=plot_or_no, plot_axis=ax_plot, plot_path=plot_path, plot_name=plot_name)

            # print('self.num_peak_deleted', self.num_peak_deleted)

            # Fill the gap with spline
            plot_name = rigidbody + '_fill_gap_trash_' + ax_plot
            plot_path = self.trial_folder_path + 'Vicon/' + plot_name + '.png'
            self.df_vicon_filtered.loc[:, (rigidbody, position_iterables)] = analysis.fill_gap_trash(
                self.df_vicon_filtered.loc[:, (rigidbody, position_iterables)].to_numpy(), kind='linear', plotit=plot_or_no, plot_axis=ax_plot, plot_path=plot_path, plot_name=plot_name)

            # Filter the data with butterworth filter
            plot_name = rigidbody + '_filt_my_signal_' + ax_plot
            plot_path = self.trial_folder_path + 'Vicon/' + plot_name + '.png'
            for ax in position_iterables:
                if rigidbody in rigidbody_list_plot and ax == ax_plot:
                    plot_or_no = True
                else:
                    plot_or_no = False

                self.df_vicon_filtered.loc[:, (rigidbody, ax)] = analysis.filt_my_signal(
                    self.df_vicon_filtered.loc[:, (rigidbody, ax)], freq=120, lp=7, n=4, plotit=plot_or_no, plot_axis=ax_plot, plot_path=plot_path, plot_name=plot_name)

        # Save filtered vicon dataframe to csv file
        self.df_vicon_filtered.to_csv(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_filtered.csv', sep=';')

        # Save filtered vicon dataframe to pickle file
        self.df_vicon_filtered.to_pickle(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_filtered.pkl')

        return True

    # Export cs3 file with vicon data
    def export_vicon_to_c3d(self):
        if not self.load_vicon_filtered():
            return
        if not self.load_vicon_structured():
            return
        points_filtered = []
        points_non_filtered = []

        for rigidbody in self.list_rigidbody_in_trial:
            point_filtered = self.df_vicon_filtered.loc[:, (rigidbody, position_iterables)].to_numpy()
            point_non_filtered = self.df_vicon_structured.loc[:, (rigidbody, position_iterables)].to_numpy()
            # Bug in c3d: the x, y, z axis is reversed, so we have to apply reverse to compensate
            point_filtered = np.hstack((point_filtered, np.zeros((point_filtered.shape[0], 2), dtype=point_filtered.dtype)))
            point_non_filtered = np.hstack((point_non_filtered, np.zeros((point_non_filtered.shape[0], 2), dtype=point_non_filtered.dtype)))
            points_filtered.append(point_filtered)
            points_non_filtered.append(point_non_filtered)

        for marker in self.list_marker_in_trial:
            point_filtered = self.df_vicon_filtered.loc[:, (marker, position_iterables)].to_numpy()
            point_non_filtered = self.df_vicon_structured.loc[:, (marker, position_iterables)].to_numpy()
            # Bug in c3d: the x, y, z axis is reversed, so we have to apply reverse to compensate
            point_filtered = np.hstack((point_filtered, np.zeros((point_filtered.shape[0], 2), dtype=point_filtered.dtype)))
            point_non_filtered = np.hstack((point_non_filtered, np.zeros((point_non_filtered.shape[0], 2), dtype=point_non_filtered.dtype)))
            points_filtered.append(point_filtered)
            points_non_filtered.append(point_non_filtered)

        points_filtered = np.array(points_filtered)
        points_non_filtered = np.array(points_non_filtered)

        if len(points_filtered) < 1:
            return

        # Write filtered c3d
        writer = c3d.Writer(point_rate=120, point_units='mm  ', point_scale=-1.)
        list_rigidbody_and_marker = self.list_rigidbody_in_trial + self.list_marker_in_trial
        writer.set_point_labels(list_rigidbody_and_marker)
        for i in range(points_filtered.shape[1]):  # for each frame
            writer.add_frames([(points_filtered[:, i], np.array([[]]))])  # empty 2 dimensional ndarray for analogs
        with open(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_' + 'filtered' + '.c3d', 'wb') as h:
            writer.write(h)
        c3d_writer_new_lib = ezc3d.c3d(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_' + 'filtered' + '.c3d')
        c3d_writer_new_lib.write(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_' + 'filtered' + '.c3d')

        # Write non filtered c3d
        writer = c3d.Writer(point_rate=120, point_units='mm  ', point_scale=-1.)
        list_rigidbody_and_marker = self.list_rigidbody_in_trial + self.list_marker_in_trial
        writer.set_point_labels(list_rigidbody_and_marker)
        for i in range(points_non_filtered.shape[1]):  # for each frame
            writer.add_frames([(points_non_filtered[:, i], np.array([[]]))])  # empty 2 dimensional ndarray for analogs
        with open(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_' + 'structured' + '.c3d', 'wb') as h:
            writer.write(h)
        c3d_writer_new_lib = ezc3d.c3d(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_' + 'structured' + '.c3d')
        c3d_writer_new_lib.write(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_' + 'structured' + '.c3d')

    # Load the raw data vicon dataframe from the pickle file
    def load_vicon_raw(self):
        if self.df_vicon_raw is not None:
            return True
        try:
            self.df_vicon_raw = pd.read_pickle(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_raw.pkl')
            if len(self.list_rigidbody_in_trial) == 0:
                self.extract_column_information_vicon()
            return True
        except FileNotFoundError:
            print('The vicon_raw.pkl file for ' + self.config.name + ' in session ' + self.session.name + ' does not exist!')
            return False

    # Load the structured vicon dataframe from the pickle file
    def load_vicon_structured(self):
        if self.df_vicon_structured is not None:
            return True
        try:
            self.df_vicon_structured = pd.read_pickle(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_structured.pkl')
            if len(self.list_rigidbody_in_trial) == 0:
                self.extract_column_information_vicon()
            return True
        except FileNotFoundError:
            print('The vicon_structured.pkl file for ' + self.config.name + ' in session ' + self.session.name + ' does not exist!')
            return False

    # Load the filtered vicon dataframe from the pickle file
    def load_vicon_filtered(self):
        if self.df_vicon_filtered is not None:
            return True
        try:
            self.df_vicon_filtered = pd.read_pickle(self.trial_folder_path + 'Vicon/' + self.config.name + '_vicon_filtered.pkl')
            if len(self.list_rigidbody_in_trial) == 0:
                self.extract_column_information_vicon()
            return True
        except FileNotFoundError:
            print('The vicon_filtered.pkl file for ' + self.config.name + ' in session ' + self.session.name + ' does not exist!')
            return False

    # Export the trial result
    def export_trial_result(self):
        self.result.num_peak_deleted = self.num_peak_deleted
        self.result.generate_trial_result()


# The trial configuration
class TrialConfig:
    def __init__(self):
        # Handled in child class
        pass


# The instance that contains all results for a trial
class TrialResult:
    def __init__(self, trial_):
        self.trial = trial_
        self.trial_duration = None  # in second
        self.walking_duration = None  # in second
        self.walking_distance = None  # in millimeter
        self.walking_speed = None  # in millimeter/second
        self.sd_walking_speed = None  # the standard deviation of walking speed
        self.pitch_head_mean = None  # the mean pitch head orientation using degree
        self.pitch_head_sd = None  # the standard deviation pitch head orientation using degree
        self.num_peak_deleted = 0  # the number of peak detected and deleted for the main rigidbody
        self.num_abnormal_value = 0  # the number of abnormal value for the trial

        # The dictionary of global variables for the trial
        self.dic_global_variables = None

        # The trial result dataframe
        self.df_trial_result = None

    # Create a result dataframe starting with all elements in the trial config
    def initialize_trial_result(self):
        self.df_trial_result = pd.DataFrame(columns=[*self.trial.config.list_config_elements], index=range(1))
        # Feed the config data into the result dataframe
        for key, item in self.trial.config.list_config_elements.items():
            self.df_trial_result[key] = item
        # Add Nexus column
        self.df_trial_result["Post_Nexus"] = [False] * self.df_trial_result.shape[0]

    # Generate the dictionary of global variables result for the trial
    def generate_global_variables_result(self):
        self.dic_global_variables = {'trial_duration': self.trial_duration,
                                     'walking_duration': self.walking_duration,
                                     'walking_distance': self.walking_distance,
                                     'walking_speed': self.walking_speed,
                                     'sd_walking_speed': self.sd_walking_speed,
                                     'pitch_head_mean': self.pitch_head_mean,
                                     'pitch_head_sd': self.pitch_head_sd,
                                     'num_peak_deleted': self.num_peak_deleted,
                                     'num_abnormal_value': self.num_abnormal_value,
                                     }
        # Feed the global variables into the result dataframe
        for key, item in self.dic_global_variables.items():
            self.df_trial_result[key] = item

    # Generate trial result data for global variables and spatial temporal variables for each crossing
    def generate_trial_result(self):
        self.initialize_trial_result()
        self.generate_global_variables_result()
        # The rest is handled in child class
