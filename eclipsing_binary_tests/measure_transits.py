import os 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk

   
def count_sectors(val):
    return len(val.split(","))

def load_ebs():
    ebs = pd.read_csv(
        "../data/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv", 
        index_col=0
        ).dropna(subset="sectors")
    
    ebs["n_sectors"] = ebs.sectors.apply(count_sectors)
    cvzs = ebs.query("n_sectors > 10").sort_index()
    return cvzs


if __name__ == "__main__":
    pass