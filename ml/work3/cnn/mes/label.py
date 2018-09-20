#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd

labels = pd.read_table("train.txt", delim_whitespace=True, header=None, names=["File","Label"])
augmentation = pd.read_table("augmentation.txt")
dic = pd.Series(labels.Label.values,index=labels.File).to_dict()

df = {}

for row in augmentation.File:
    try:
        df[row] = dic[row[14:24]]
    except:
        pass

print (len(df))

out = pd.DataFrame(list(df.items()), columns=["File","Label"])
out.to_csv(r"pandas.txt", header=None, index=None, sep=' ', mode='w')
# print (out)
