import os
import random
import logging
import numpy as np
from tqdm import tqdm


class NTU_Reader():
    def __init__(self, dataset_root_folder, case="view", num_class=60, dataset="ntu60",
                 take_2_persons=True, get_distribution=False):
        
        self.begin_path = dataset_root_folder
        self.case = case
        if num_class > 2:
            self.num_class = num_class
        else:
            raise ValueError(f"Number of classes ({num_class}) should be >= 2 .")
        self.dataset = dataset
        self.take_2_persons = take_2_persons
        self.get_distribution = get_distribution

    def classes_we_have(self,npy_datalist,exclude_class):
        """
            This function finds all the unique classes in our dataset
            npy_datalist: an array of strings of the name of files in our dataset
            exclude_class: classes we wish to exclude from our dataset.

            returns
                - class list 0 to max_class
                - data samples of class in [0,max_class]
        """
        max_class = self.num_class
        #Here is a simpler implementation
        arr = [element for element in npy_datalist if int(element[17:20]) not in exclude_class] #This removes unwanted classes
        the_class = [int(ele[17:20]) for ele in arr] #This gets the default classes (with repetitions)
        the_class = set(the_class) #This removes repetitions
        the_class = list(the_class) #Converts back to a list
        the_class.sort() #Sorts it before returning
        #Reduce the classes further
        arr_p = [element for element in arr if int(element[17:20]) in the_class[:max_class]]

        return the_class[:max_class], arr_p


    def get_0_to_max_class(self,classes):
        """
            classes is a sorted list
            This function maps all the classes to range(0,len(classes))
            
            info:
                - This is useful when we are excluding some classes (especially when considering NTU120).
        """
        max_class = len(classes)
        arr = np.ones((max_class,2))
        arr[:,1] = list(range(max_class))
        arr[:,0] = classes
        return arr


    def get_dataset_partisions(self):

        if self.case == "subject":
            #subjects count: 106
            # x_subjects = list(range(1,107))
            x_subjects_train = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
            28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
            70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]

            return x_subjects_train

            # x_subjects_test = list(set(x_subjects).difference(x_subjects_train))

        elif self.case == "setup":
            #set up count: 32
            x_setup_train = list(range(2,33,2)) # train (even)

            return x_setup_train
            
        elif self.case == "view":
            #camera view --> 1: +45^deg, 2: 0^deg, 3: -45^deg
            x_views_train = [2,3]
            # x_views_test = [1]

            return x_views_train
        
        else:
            raise ValueError(f"cases are: view, subject and setup. {self.case} is a wrong option.")


    def set_class(self, data_ent, mapped_class):
        """
            data_ent is a dictionary
            mapped_class is the result of self.get_0_to_max_class

            info:
                - This is useful when we use just a part of the classes
                - and need to order the class again [0, max_class].
        """
        file_name = 'file_name'
        default_class = int(data_ent[file_name][17:20])
        indx = np.where(mapped_class[:,0] == default_class)[0][0]
        new_class = mapped_class[indx,1]
        data_ent['class'] = new_class


    def get_view_or_subj(self, data, s_V_p): # s_V_p: setup, view or person(subject)
        """This function returns the relevant ID of the sample, for either subject, view or setup."""

        try:
            remap = {"subject": 'P', "view": 'V', "setup": 'S'}
        except KeyError as e:
            print(f'case should either be view, subject or setup.')
            raise e

        s_V_p = remap[s_V_p]

        file_name = 'file_name'
        name = data[file_name]
        if s_V_p=='S': #setup
            what_we_want = int(name[1:4])
        elif s_V_p=='V': #camera/view
            what_we_want = int(name[5:8])
        elif s_V_p=='P': #person/subject
            what_we_want = int(name[9:12])

        return what_we_want

    def gendata(self):
        """
            This function combines all the process that provides us a suitable data to work with.
            small=True means only NTU-RGB D60 is considered
            max_class (type: int) is the maximum class we want. Set this to 0  take all 120 classes.
        """
        
        #These are the paths containing the numpy data we want
        the_path_60 = os.path.join(self.begin_path,'raw_npy60')
        npy_datalist_60 = os.listdir(the_path_60)
        npy_datalist = npy_datalist_60

        if self.dataset.lower() == "ntu120":
            the_path_120 = os.path.join(self.begin_path,'raw_npy120')
            npy_datalist_120 = os.listdir(the_path_120)
            npy_datalist.extend(npy_datalist_120)

        if not self.take_2_persons:
            exclude_class = list(range(50,61)) + list(range(106,121)) #The action classes involving 2 persons.
            print("REMOVING UNWANTED CLASSES...")
        else:
            exclude_class = []
        
        default_classes, npy_datalist = self.classes_we_have(npy_datalist,exclude_class) #Default classes of each sample in our dataset
        mapped_classes = self.get_0_to_max_class(default_classes) #maps the classes to range(0,len(default_classes))

        print(f"\nSize of dataset: {len(npy_datalist)}")
        print(f"Max class is {self.num_class}.")

        x_train = self.get_dataset_partisions()
        
        cross_x_train, cross_x_test = [], []

        for each in tqdm(npy_datalist):

            if each in npy_datalist_60:
                holder = np.load(os.path.join(the_path_60,each),allow_pickle=True).item()
                self.set_class(holder,mapped_classes)
                
                # put in subject or setup or view
                if self.get_view_or_subj(holder,self.case) in x_train:
                    cross_x_train.append(holder)
                else:
                    cross_x_test.append(holder)

            else:
                if self.dataset.lower() != "ntu120":
                    continue #ensures it skips when we consider only dataset 60
                
                holder = np.load(os.path.join(the_path_120,each),allow_pickle=True).item()
                self.set_class(holder,mapped_classes)

                # put in subject or setup or view
                if self.get_view_or_subj(holder,self.case) in x_train:
                    cross_x_train.append(holder)
                else:
                    cross_x_test.append(holder)
        
        print()
        if self.case == "subject":
            print("\n----Returning Cross-subject split----")
        elif self.case == "setup":
            print("\n----Returning Cross-setup split----")
        elif self.case == "view":
            print("\n----Returning Cross-view split----")
        
        return np.array(cross_x_train), np.array(cross_x_test)
    

    def save_data(self, x_train, x_test):
        train_file = os.path.join(self.begin_path,f"cross_{self.case}_train.npy")
        test_file = os.path.join(self.begin_path,f"cross_{self.case}_test.npy")

        if len(x_train) != 0:
            print(f"Saving cross-{self.case} data...")
            print(f"Train data size: {len(x_train)}")
            print(f"Test data size: {len(x_test)}")

            with open(train_file,'wb') as f:
                np.save(f,x_train)
            with open(test_file,'wb') as f:
                np.save(f,x_test)

        
    def start(self):    
        logging.info(f'Phase: Train and Test data at once.')
        x_train, x_test = self.gendata()
        self.save_data(x_train, x_test)
        

    def normalise_data(self, pose_data, random_idx=False):
        """This function normalises the skeleton data."""

        pass