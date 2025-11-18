#!/usr/bin/env python
# coding=utf-8

'''
transform the skeleton data in NTU RGB+D dataset into the numpy arrays for a more efficient data loading
'''
import argparse
import numpy as np
import os
import sys

user_name = 'olayinkaajayi'
#We can use step_ranges to set the classes we skip while processing (I am yet to do this).
step_ranges = list(range(0,100)) # just parse range, for the purpose of parallel running.


def _print_toolbar(rate, annotation=''):
    toolbar_width = 50
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def _end_toolbar():
    sys.stdout.write('\n')

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True
    return missing_files

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4 #The maximum number of bodies we want in each frame.
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the
    # abundant bodys.
    # read all lines into the pool to speed up, less io operation.
    nframe = int(datas[0][:-1]) #The entries are a strings so we take all the values on each line and convert to type int. The -1 in [:-1] helps avoid the '\n'(newline) character
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9] #Pick all characters starting at S00**********, just before .skeleton
    nbody = int(datas[1][:-1]) #The entries are a strings so we take all the values on each line and convert to type int. The -1 in [:-1] helps avoid the '\n'(newline) character
    bodymat['nbodys'] = []
    bodymat['njoints'] = njoints
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        #We go through each frame and construct the skeleton data in each.
        cursor += 1 #1
        bodycount = int(datas[cursor][:-1]) #The entries are a strings so we take all the values on each line and convert to type int. The -1 in [:-1] helps avoid the '\n'(newline) character
        if bodycount == 0:
            continue
        # skip the empty frame
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            #For each body in the frame, we construct the skeleton.
            cursor += 1 #2
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)

            #This variable was not used
            bodyinfo = datas[cursor][:-1].split(' ') #Expect this to be an array or a single string which we partition into an array using split().
            cursor += 1 #3

            njoints = int(datas[cursor][:-1]) #Expect this to be a scalar.
            for joint in range(njoints):
                #For each joint in each body, we provide the information there in.
                cursor += 1 #4
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo))) #jointinfo is of length 7 (and the first 3 entries are the x,y,z component of each joint)
                if save_skelxyz:
                    bodymat[skel_body][frame,joint] = jointinfo[:3] #Recall that we already initialized this as bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
                if save_depthxy:
                    bodymat[depth_body][frame,joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame,joint] = jointinfo[5:7] #jointinfo has the following entries (camera(x,y,z),colour(x,y),depth(x,y),quaterion(x,y,z,w)) in this order.
    # prune the abundant bodys
    for each in range(max_body):
        if each >= max(bodymat['nbodys']): #This is fine
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
    return bodymat

############## Split data into the different splits (cross-view): Friday, 14th November 2025


def main():
    parser = argparse.ArgumentParser(
        description='This code is meant to give us our data in the format we want and specify the number of classes.')
    parser.add_argument('--begin_path', type=str, default='./data',
                        help='parent path that leads to DHCS_implement (default: /h/ola/Docs/Gith/mthSys)')
    args = parser.parse_args()

    begin_path = args.begin_path


    #Note whether you wish to do this for NT-RGB-D60 or D120
    save_npy_path = os.path.join(begin_path,'raw_npy60')
    if not os.path.exists(save_npy_path):
        os.mkdir(save_npy_path)
    load_txt_path = os.path.join(begin_path,'raw_txt60')
    missing_file_path = os.path.join(begin_path,'ntu_rgb60_missings.txt') #3 lines containing sentences removed.


    missing_files = _load_missing_file(missing_file_path)
    datalist = os.listdir(load_txt_path) #returns the name of all files in the path
    alread_exist = os.listdir(save_npy_path)
    alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))

    cnt = 0 #count how many files in missing skeleton
    for ind, each in enumerate(datalist):
        _print_toolbar(ind * 1.0 / len(datalist),
                       '({:>5}/{:<5})'.format(
                           ind + 1, len(datalist)
                       ))
        S = int(each[1:4])
        # if S not in step_ranges: #This will exclude some skeletons: I do not want it!!!!!!
        #     continue
        if each+'.skeleton.npy' in alread_exist_dict:
            print('file already existed !')
            continue
        if each[:20] in missing_files:
            cnt += 1
            print('file missing:',cnt)
            continue
        #####****I can always add a code here to skip specific classes****#######(Will do that in process_skeleton.py)
        loadname = load_txt_path+each
        # print(each) #Remove this
        mat = _read_skeleton(loadname,save_rgbxy=False,save_depthxy=False) #set save_rgbxy & save_depthxy as false since I would not need the values
        mat = np.array(mat)
        save_path = save_npy_path+'{}.npy'.format(each)
        np.save(save_path, mat)
        # raise ValueError()
    _end_toolbar()

if __name__ == '__main__':
    main()
