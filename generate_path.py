"""
This script is used to generate the dataset path to pytorch.
-----------------------
Training set: 1, 2, 3, 4, 5, 6, 9
Validation set: 7, 8
Test set: 10
"""


for case in range(1, 11):
    for i in range(7, 20):
        if case == 10:
            filename = 'test_list_whole.txt'
        else:
            filename = 'train_list_whole.txt'
        if case == 7 or case == 8:
            filename = 'val_list_whole.txt'
        with open(filename, 'a') as f:
            f.write("./data/case{0}/ct1_{1}.gif ./data/case{0}/mr2_{1}.gif\n".format(case, str(i).zfill(3)))
            
            
version = 'my_result'
for case in range(1, 11):
    for i in range(7, 20):
        if case == 10:
            filename = version + '_test_list_whole.txt'
        else:
            filename = version + '_train_list_whole.txt'
        if case == 7 or case == 8:
            filename = version + '_val_list_whole.txt'
        with open(filename, 'a') as f:
            f.write("./my_result/{0}_{1}.tif ./data/case{0}/ct1_{1}.gif ./data/case{0}/mr2_{1}.gif\n".format(case, str(i).zfill(3)))


for case in range(1, 11):
    for i in range(0, 24):
        filename = 'final_all_data.txt'
        with open(filename, 'a') as f:
            f.write("./data/case{0}/ct1_{1}.gif ./data/case{0}/mr2_{1}.gif\n".format(case, str(i).zfill(3)))