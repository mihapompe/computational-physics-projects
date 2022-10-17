import core
from visuals import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import optimize 
import pandas as pd
import re


def clean_data(length):
    clean_data = []
    with open(f"38400_baud_{length}.txt", "r") as f:
        data = f.read()
    pattern = r"[0-9]{2}:([0-9]{2}):([0-9]{2}).([0-9]{3}) -> ([0-9]{3})\t([0-9]{1,3})\n\n"
    new_pattern = r"\1 \2 \3 \4 \5\n"
    data = re.sub(pattern, new_pattern, data)
    data = data.split("\n")
    for line in data[:-1]:
        entry = []
        comp = line.split(" ")
        entry.append(int(comp[0])*60+int(comp[1])+int(comp[2])/1000)
        entry.append(int(comp[3]))
        entry.append(int(comp[4]))
        clean_data.append(entry)

    data = np.array(clean_data)  
    data[:,0] -= data[0,0]
    return data.T
