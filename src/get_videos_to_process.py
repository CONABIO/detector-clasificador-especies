#!/usr/bin/env python3
"""

input:
    ${NUM_PROCESSES}
    ${VIDEOS_JSON_DIR}
    ${CSV_DIR}
    ${MODE}
"""
import pandas as pd
from camtraproc.utils.utils import query_s3, query_irekua, query_manual
import os
import sys

num_processes = int(sys.argv[1])
all_info_dir = sys.argv[2]
csv_dir = sys.argv[3]
mode = sys.argv[4]
pl_batches_size = int(sys.argv[5])

_ = [os.makedirs(os.path.join(csv_dir,str(f)), exist_ok=True) for f in range(1,num_processes+1)]

if mode == 's3':
    df = query_s3('snmb','videos_fauna',r'.*AVI$',all_info_dir)
elif mode == 'irekua':
    df = query_irekua(all_info_dir)
elif mode == 'manual':
    df = query_manual(all_info_dir)
else:
    print('Invalid mode')

    
if 'sequence_id' in df.keys():
    if len(df['sequence_id'].drop_duplicates()) < num_processes:
        df.to_csv(os.path.join(csv_dir,'1','1_files_coor.csv'), index=False )

    elif len(df['sequence_id'].drop_duplicates()) < pl_batches_size*num_processes:
        batches_size = len(df['sequence_id'].drop_duplicates())//num_processes
        dfl = [pd.DataFrame(y) for x, y in df.groupby(by=['sequence_id'], as_index=False)]
        _ = [ pd.concat(dfl[(i-1)*batches_size:(i)*batches_size]).to_csv(os.path.join(csv_dir,str(i),'1_files_coor.csv'),index=False) 
             for i in range(1,num_processes+1) ]
        if len(df['sequence_id'].drop_duplicates()) % num_processes:
            pd.concat(dfl[num_processes*batches_size:]).to_csv(os.path.join(csv_dir,'1','2_files_coor.csv'),index=False)
    else:
        steps = len(df['sequence_id'].drop_duplicates())//(pl_batches_size*num_processes)
        dfl = [pd.DataFrame(y) for x, y in df.groupby(by=['sequence_id'], as_index=False)]
        _ = [pd.concat(dfl[(s-1)*pl_batches_size*num_processes+(i-1)*pl_batches_size:(+s-1)*pl_batches_size*num_processes+i*pl_batches_size]).to_csv(os.path.join(csv_dir,str(i),'{}_files_coor.csv'.format(s)),index=False) 
             for s in range(1,steps+1) for i in range(1,num_processes+1) ]

        batches_size = (len(df['sequence_id'].drop_duplicates())-steps*pl_batches_size*num_processes)//num_processes
        if batches_size < 1:
            pd.concat(dfl[steps*pl_batches_size*num_processes:]).to_csv(os.path.join(csv_dir,'1','{}_files_coor.csv'.format(steps+1)),index=False)
        else:
            _ = [pd.concat(dfl[steps*pl_batches_size*num_processes+(i-1)*batches_size:steps*pl_batches_size*num_processes+(i)*batches_size]).to_csv(os.path.join(csv_dir,str(i),'{}_files_coor.csv'.format(steps+1)),index=False) 
                 for i in range(1,num_processes+1) ]
            if (len(df['sequence_id'].drop_duplicates())-steps*pl_batches_size*num_processes) % num_processes:
                pd.concat(dfl[steps*pl_batches_size*num_processes+num_processes*batches_size:]).to_csv(os.path.join(csv_dir,'1','{}_files_coor.csv'.format(steps+2)),index=False)

else:
    if len(df) < num_processes:
        df.to_csv(os.path.join(csv_dir,'1','1_files_coor.csv'), index=False )

    elif len(df) < pl_batches_size*num_processes:
        batches_size = len(df)//num_processes
        _ = [df.iloc[(i-1)*batches_size:(i)*batches_size].reset_index().drop(['index'], axis=1).to_csv(os.path.join(csv_dir,str(i),'1_files_coor.csv'),index=False) 
             for i in range(1,num_processes+1) ]
        if len(df) % num_processes:
            df.iloc[num_processes*batches_size:].reset_index().drop(['index'], axis=1).to_csv(os.path.join(csv_dir,'1','2_files_coor.csv'),index=False)
    else:
        steps = len(df)//(pl_batches_size*num_processes)
        _ = [df.iloc[(s-1)*pl_batches_size*num_processes+(i-1)*pl_batches_size:(+s-1)*pl_batches_size*num_processes+i*pl_batches_size].reset_index().drop(['index'], axis=1).to_csv(os.path.join(csv_dir,str(i),'{}_files_coor.csv'.format(s)),index=False) 
             for s in range(1,steps+1) for i in range(1,num_processes+1) ]

        batches_size = (len(df)-steps*pl_batches_size*num_processes)//num_processes
        if batches_size < 1:
            df.iloc[steps*pl_batches_size*num_processes:].reset_index().drop(['index'], axis=1).to_csv(os.path.join(csv_dir,'1','{}_files_coor.csv'.format(steps+1)),index=False)
        else:
            _ = [df.iloc[steps*pl_batches_size*num_processes+(i-1)*batches_size:steps*pl_batches_size*num_processes+(i)*batches_size].reset_index().drop(['index'], axis=1).to_csv(os.path.join(csv_dir,str(i),'{}_files_coor.csv'.format(steps+1)),index=False) 
                 for i in range(1,num_processes+1) ]
            if (len(df)-steps*pl_batches_size*num_processes) % num_processes:
                df.iloc[steps*pl_batches_size*num_processes+num_processes*batches_size:].reset_index().drop(['index'], axis=1).to_csv(os.path.join(csv_dir,'1','{}_files_coor.csv'.format(steps+2)),index=False)
