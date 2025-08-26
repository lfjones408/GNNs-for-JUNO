import os
import numpy as np
import pandas as pd

fileList  = ['nu_mu', 'antinu_mu',]
filePaths = '/hepstore/ljones/atm_nu/J24.1.2/FC/'
summary   = '/log/performance_summary.csv'

files = []

for file in fileList:
    print(file)
    path = filePaths + file + summary
    df = pd.read_csv(path)

    jobs = df['JobID']
    events  = df['EventsSaved']
    num_events = 0

    for job, evt in zip(jobs, events):
        num_events += evt

        if(num_events < 20000):
            print(f'    Job -> {job} \n         nEvents -> {evt}')
            files.append(filePaths + file + f'/pmt_data_{job}.h5\n')
        else:
            continue

txt_file = open("training_files_nu_mu.txt", "w")
txt_file.writelines(files)
txt_file.close()

print('Data Written!')