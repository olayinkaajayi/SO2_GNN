import argparse
import numpy as np
import sys
sys.path.append("DHCS_implement/data_cleaning") #would be needed
from process_skeleton_data import get_data


def main():
    parser = argparse.ArgumentParser(
        description='This code is meant to give us our data in the format we want and specify the number of classes.')
    parser.add_argument('--begin_path', type=str, default="/dcs/large/u2034358/",
                        help='parent path that leads to DHCS_implement (default: ~/codes)')
    parser.add_argument('--py_v', type=str, default="py3",
                        help='whether python 2 or 3 (default: py2')
    parser.add_argument('--max-class', dest="max_class", type=int, default=49,
                        help='specifies the maximum class (default: 60)')
    parser.add_argument('--two-persons', dest='take_2p', action='store_true', default=False,
                        help='decide if we consider actions involving 2 persons')
    parser.add_argument('--view', dest='cross_view', action='store_true', default=False,
                        help='when we want the cross-view data partition')
    parser.add_argument('--sub', dest='cross_subject', action='store_true', default=False,
                        help='when we want the cross-subject data partition')
    parser.add_argument('--set', dest='cross_setup', action='store_true', default=False,
                        help='when we want the cross-setup data partition')
    parser.add_argument('--only-60', dest='only_60', action='store_true', default=False,
                        help='when we want only the data up to class 60.')

    args = parser.parse_args()

    begin_path = args.begin_path
    ##########LOAD DATA#############
    #max_class (type: int) is the maximum class we want.
    cross_sub_train, cross_sub_test, cross_view_train, cross_view_test, A, max_class = get_data(begin_path, args, max_class=args.max_class, py_v=args.py_v, take_2_persons=args.take_2p) #returns data and adjacency matrix
    ##########LOAD DATA#############

    cross_sub_train_name = 'cross_sub_train_py2' if args.py_v == 'py2' else 'cross_sub_train'
    cross_sub_test_name = 'cross_sub_test_py2' if args.py_v == 'py2' else 'cross_sub_test'
    cross_view_train_name = 'cross_view_train_py2' if args.py_v == 'py2' else 'cross_view_train'
    cross_view_test_name = 'cross_view_test_py2' if args.py_v == 'py2' else 'cross_view_test'
    adj_file_name = 'adj_matrix_py2' if args.py_v == 'py2' else 'adj_matrix'
    max_class_file_name = 'max_class_py2' if args.py_v == 'py2' else 'max_class'

    cross_sub_train_file = begin_path+cross_sub_train_name+'.npy'
    cross_sub_test_file = begin_path+cross_sub_test_name+'.npy'
    cross_view_train_file = begin_path+cross_view_train_name+'.npy'
    cross_view_test_file = begin_path+cross_view_test_name+'.npy'
    # data_file = begin_path+'/DHCS_implement/clean_data.npy'
    adj_file = begin_path+adj_file_name+'.npy'
    max_class_file = begin_path+max_class_file_name+'.npy'

    # cross views and subjects saving...
    if len(cross_sub_train) != 0:
        print("Saving cross-subject data...")
        print(f"Train data size: {len(cross_sub_train)}")
        print(f"Test data size: {len(cross_sub_test)}")
        with open(cross_sub_train_file,'wb') as f:
            np.save(f,cross_sub_train)
        with open(cross_sub_test_file,'wb') as f:
            np.save(f,cross_sub_test)

    if len(cross_view_train) != 0:
        if args.cross_view:
            print("Saving cross-view data...")
        else:
            print("Saving cross-setup data...")
        print(f"Train data size: {len(cross_view_train)}")
        print(f"Test data size: {len(cross_view_test)}")
        with open(cross_view_train_file,'wb') as f:
            np.save(f,cross_view_train)
        with open(cross_view_test_file,'wb') as f:
            np.save(f,cross_view_test)

    # Adjacency matrix and max class saving...
    with open(adj_file,'wb') as f:
        np.save(f,A)
    with open(max_class_file,'wb') as f:
        np.save(f,max_class)

if __name__ == '__main__':
    main()
    print("Files saved and ready for use.")
