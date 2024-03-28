import numpy as np
import csv
import os

OUTPUT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

label_path = OUTPUT_DIRECTORY + "/class_labels_indices.csv" 
with open(label_path) as f:
    reader = csv.reader(f)
    lst = list(reader)


lst = np.array(lst)
lst = lst[:,2]
lst = np.delete(lst,(0),axis=0)

np.save('classes.npy', lst)






