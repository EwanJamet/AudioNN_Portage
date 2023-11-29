import numpy as np
import csv
label_path = "/home/ewan/Desktop/audioset_tagging_cnn/metadata/class_labels_indices.csv" 
with open(label_path) as f:
    reader = csv.reader(f)
    lst = list(reader)


lst = np.array(lst)
lst = lst[:,2]
lst = np.delete(lst,(0),axis=0)

np.save('classes.npy', lst)






