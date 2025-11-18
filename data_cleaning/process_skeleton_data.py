"""
    This module would be used for processing the skeleton data
"""

import numpy as np
import sys
import os
sys.path.append("DHCS_implement/data_cleaning") #would be needed

from txt2npy import _print_toolbar,_end_toolbar #so I can see the progress of the code


#Setting the classes we have in our dataset
def classes_we_have(npy_datalist,exclude_class,max_class):
    """
        This function finds all the unique classes in our dataset
        npy_datalist is an array of strings of the name of files in our dataset
        exclude_class are the classes we wish to exclude from our dataset.
    """
    #Here is a simpler implementation
    arr = [element for element in npy_datalist if int(element[17:20]) not in exclude_class] #This removes unwanted classes
    the_class = [int(ele[17:20]) for ele in arr] #This gets the default classes (with repetitions)
    the_class = set(the_class) #This removes repetitions
    the_class = list(the_class) #Converts back to a list
    the_class.sort() #Sorts it before returning
    #Reduce the classes further
    arr_p = [element for element in arr if int(element[17:20]) in the_class[:max_class]]
    return the_class[:max_class],arr_p

def get_0_to_max_class(classes):
    """
        classes is a sorted list
        This function maps all the classes to range(0,len(classes))
    """
    row = len(classes)
    arr = np.ones((row,2))
    arr[:,1] = list(range(row))
    arr[:,0] = classes
    return arr,row


def get_dataset_partisions():
    #subject (106)
    x_subjects = list(range(1,107))
    x_subjects_train = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
    28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
    70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
    # x_subjects_test = list(set(x_subjects).difference(x_subjects_train))

    #set up (32)
    x_setup = list(range(1,33))
    x_setup_train = [i for i in x_setup if i%2==0] #train (even)

    #camera view
    x_views_train = [2,3]
    x_views_test = [1]

    return x_subjects_train, x_views_train, x_setup_train


def set_class(data_ent,mapped_class):
    """
        data_ent is a dictionary
        mapped_class is the result of get_0_to_max_class
    """
    file_name = 'file_name'
    default_class = int(data_ent[file_name][17:20])
    indx = np.where(mapped_class[:,0] == default_class)[0][0]
    new_class = mapped_class[indx,1]
    data_ent['class'] = new_class


def get_view_or_subj(data,s_V_p='S'): # s_V_p: setup, view oe person(subject)
    file_name = 'file_name'
    name = data[file_name]
    if s_V_p=='S': #setup
        what_we_want = int(name[1:4])
    elif s_V_p=='V': #camera/view
        what_we_want = int(name[5:8])
    elif s_V_p=='P': #person/subject
        what_we_want = int(name[9:12])
    else:
        assert s_V_p in 'SVP', 'Option is either: S, V or P.'

    return what_we_want

def get_data(begin_path, args,max_class=60,py_v='py2',take_2_persons=False):
    """
        This function combines all the process that provides us a suitable data to work with.
        small=True means only NTU-RGB D60 is considered
        max_class (type: int) is the maximum class we want. Set this to 0  take all 120 classes.
    """
    if max_class > 60:
        args.cross_view = False
    #These are the paths containing the numpy data we want
    if py_v == 'py2':
        the_path_60 = begin_path+'/raw_npy60_py2'
        the_path_120 = begin_path+'/raw_npy120_py2'
    else:
        the_path_60 = begin_path+'/raw_npy60'
        the_path_120 = begin_path+'/raw_npy120'
    npy_datalist_60 = os.listdir(the_path_60)
    npy_datalist_120 = os.listdir(the_path_120)

    if args.only_60:
        npy_datalist = npy_datalist_60
    else:
        npy_datalist = npy_datalist_60 + npy_datalist_120

    if max_class < 2: #This helps reduce the number of classes we would be considering. Making the dataset smaller.
        print("Number of classes should be >= 2 !!!")
        exit()

    if not take_2_persons:
        exclude_class = list(range(50,61)) + list(range(106,121)) #The action classes involving 2 persons. We do not wnat this for our model (yet)
        print("REMOVING UNWANTED CLASSES...")
    else:
        exclude_class = []
    default_classes, npy_datalist = classes_we_have(npy_datalist,exclude_class,max_class) #Default classes of each data in our dataset
    mapped_classes,max_class = get_0_to_max_class(default_classes) #maps the classes to range(0,len(default_classes))

    print("Size of dataset: {}".format(len(npy_datalist)))
    print("Max class is {}".format(max_class))

    # data = np.zeros(len(npy_datalist),dtype=dict) #first pre-allocate memory
    count = 0

    x_subjects_train, x_views_train, x_setup_train = get_dataset_partisions()
    cross_sub_train, cross_sub_test, cross_view_train, cross_view_test = [], [], [], []

    for each in npy_datalist:
        _print_toolbar(count * 1.0 / len(npy_datalist),
                       '({:>5}/{:<5})'.format(
                           count + 1, len(npy_datalist)
                        ))
        # print(each) #Removing this would not allow the progress bar print.
        if each in npy_datalist_60:
            # data[count] = np.load(the_path_60+'/'+each,allow_pickle=True).item()
            # set_class(data[count],mapped_classes)
            holder = np.load(the_path_60+'/'+each,allow_pickle=True).item()

            set_class(holder,mapped_classes)
            # put in subject
            if args.cross_subject:
                if get_view_or_subj(holder,s_V_p='P') in x_subjects_train:
                    cross_sub_train.append(holder)
                else:
                    cross_sub_test.append(holder)

            # put in setup/view
            if args.cross_view:
                if get_view_or_subj(holder,s_V_p='V') in x_views_train:
                    cross_view_train.append(holder)
                else:
                    cross_view_test.append(holder)

            if args.cross_setup:
                if get_view_or_subj(holder,s_V_p='S') in x_setup_train:
                    cross_view_train.append(holder)
                else:
                    cross_view_test.append(holder)

        else:
            if args.only_60:
                continue #ensures it skips when we consider only dataset 60
            # data[count] = np.load(the_path_120+'/'+each,allow_pickle=True).item()
            # set_class(data[count],mapped_classes)
            holder = np.load(the_path_120+'/'+each,allow_pickle=True).item()

            set_class(holder,mapped_classes)
            # put in subject
            if args.cross_subject:
                if get_view_or_subj(holder,s_V_p='P') in x_subjects_train:
                    cross_sub_train.append(holder)
                else:
                    cross_sub_test.append(holder)

            # put in setup/view
            if args.cross_view:
                if get_view_or_subj(holder,s_V_p='V') in x_views_train:
                    cross_view_train.append(holder)
                else:
                    cross_view_test.append(holder)

            if args.cross_setup:
                if get_view_or_subj(holder,s_V_p='S') in x_setup_train:
                    cross_view_train.append(holder)
                else:
                    cross_view_test.append(holder)

        count += 1
    _end_toolbar()

    #Pre-defined matrix
    # njoints = list(data[0].keys())[1] #key name
    if args.cross_subject:
        njoints = cross_sub_train[0]['njoints'] #This should be 25
    elif args.cross_view:
        njoints = cross_view_train[0]['njoints'] #This should be 25
    elif args.cross_setup:
        njoints = cross_view_train[0]['njoints'] #This should be 25
    else:
        njoints = 25

    A = np.zeros((njoints,njoints))
    def add_connection(A,indx_a,indx_b):
        """
            When a pair of nodes have a connection,
            this function adds 1 to their position in the
            (symetric) adjacency matrix.
        """
        A[indx_a-1,indx_b-1] = 1
        A[indx_b-1,indx_a-1] = 1

    #####Create connection##############
    #Waist downward
    add_connection(A,1,13)
    add_connection(A,1,17)
    add_connection(A,17,18)
    add_connection(A,18,19)
    add_connection(A,19,20)
    add_connection(A,13,14)
    add_connection(A,14,15)
    add_connection(A,15,16)
    #Waist to the head
    add_connection(A,1,2)
    add_connection(A,2,21)
    add_connection(A,21,3)
    add_connection(A,3,4)
    #Neck to right hand
    add_connection(A,21,9)
    add_connection(A,9,10)
    add_connection(A,10,11)
    add_connection(A,11,12)
    add_connection(A,12,24)
    add_connection(A,12,25)
    #Neck to left hand
    add_connection(A,21,5)
    add_connection(A,5,6)
    add_connection(A,6,7)
    add_connection(A,7,8)
    add_connection(A,8,22)
    add_connection(A,8,23)
    ############Finish creating connections
    print("##DATA COLLECTION DONE!!!##")
    return np.array(cross_sub_train), np.array(cross_sub_test), np.array(cross_view_train), np.array(cross_view_test) , A, max_class #send the data, adjacency matrix and max_class
