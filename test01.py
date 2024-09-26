import pandas as pd
import numpy as np

data1 = np.array([['111', '222']])

data2 = pd.read_csv("train_dataset/folder1/metadata.csv")
data2 = np.array(data2)
print(data1.shape,data2.shape)

data = np.vstack((data1,data2))
data = pd.DataFrame(data, columns=['file_name', 'sentence'])
print(data)