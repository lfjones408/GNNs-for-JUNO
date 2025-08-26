import os
import glob
import numpy as np
import pandas as pd

fileList  = ['nu_mu', 'antinu_mu']
filePaths = '/hepstore/ljones/atm_nu/J24.1.2/FC/'
summary   = '/log/performance_summary.csv'

files = []

with open('training_files_nu_e.txt', 'r') as path:
        training_path = [line.strip() for line in path]

for file in fileList:
    print(file)
    path = filePaths + file + summary
    df = pd.read_csv(path)

    jobs = df['JobID']
    events  = df['EventsSaved']
    num_events = 0

    for job, evt in zip(jobs, events):
        h5_path = filePaths + file + f'/pmt_data_{job}.h5'

        if h5_path not in training_path:
            num_events += evt
            if(num_events < 20000):
                print(f'    Job -> {job} \n         nEvents -> {evt}')
                files.append(h5_path+'\n')
            else:
                continue

txt_file = open("evaluation_files_nu_e.txt", "w")
txt_file.writelines(files)
txt_file.close()

print('Data Written!')