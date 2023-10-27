import c3d
import ezc3d
import os

loco_path = 'C:/Users/lthongkh/OneDrive - JNJ/Loco/Loco_Analysed'

#session_done = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012', '1013', '1014', '1015', '1017', '1018', '1019', '1021', '1022', '1023', '1024', '1025', '1026', '1027', '2001', '2002']
session_done = []

def already_done(path):
    for session in session_done:
        if(session in path):
            return True
    return False

for type_path in os.listdir(loco_path):
    #if(type_path[-8:] == 'Analysed'):
    if(type_path == 'Crossing_Analysed'):
        for session_path in os.listdir(loco_path + '/' + type_path):
            if(not already_done(session_path)):
                data_path = loco_path + '/' + type_path + '/' + session_path + '/Data'
                for trial_path in os.listdir(data_path):
                    if("Adaptation" not in trial_path):
                        c3d_path = data_path + '/' + trial_path + '/Vicon/' + trial_path + '_vicon_structured.c3d'
                        if(os.path.exists(c3d_path)):
                            print(c3d_path)
                            c3d_writer_new_lib = ezc3d.c3d(c3d_path)
                            c3d_writer_new_lib.write(c3d_path)
            if(session_path[2:6] not in session_done):
                session_done.append(session_path[2:6])
            print(session_done)