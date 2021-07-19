#!/usr/bin/env python3
"""
Detecta fauna, humanos y posibles humanos. Sube anotaciones de humanos a irekua y tal vez pone etiqueta de posibles humanos a esos
que aunque se clasifican, podria poner etiqueta de revisar para que esas imagenes se revisen manualmente.. podria ser, para que
tambien las que rechaza en clasificacion sean revisadas por humano y no se vuelvan a procesar con ese modelo
eriqueta de procesado podria ser y otra de dudoso.
input:
"$f"'_files_coor.csv'
${BBOXES_DIR}'/'${NUM_PROCESS}
${CSV_DIR}'/'${NUM_PROCESS}
detector_file.pb
    file
    bboxes_dir
    csv_dir
    detector_file
"""
from camtraproc.detection.detection import run_megadetector, load_model
from camtraproc.detection.motionm import run_motionmeerkat
from camtraproc.settings import IMAGE_TYPE, VIDEO_TYPE, MODE
import pandas as pd
import os
import sys


csv_file = sys.argv[1]
bboxes_dir = sys.argv[2]
csv_dir = sys.argv[3]
detector_file = sys.argv[4]
num_str = csv_file.split('_')[0]

df = pd.read_csv(os.path.join(csv_dir,csv_file))
dfl = [pd.DataFrame(y) for x, y in df.groupby(by=['item_type'], as_index=False)]
#_ = [df['index2'] = df.index for df in dfl]

detection_graph4 = load_model(detector_file)

results_list = [run_megadetector(detection_graph4,bboxes_dir,df,'image',MODE) if df['item_type'].iloc[0] == IMAGE_TYPE else
     run_megadetector(detection_graph4,bboxes_dir,df,'video',MODE) if df['item_type'].iloc[0] == VIDEO_TYPE else None for df in dfl]

if None in results_list:
    raise ValueError('invalid item_type!')

try:
    dfspecies = pd.concat([dflist[0] for dflist in results_list])
    dfhumans = pd.concat([dflist[1] for dflist in results_list])
    dfmaybe_humans = pd.concat([dflist[2] for dflist in results_list])
except Exception as e:
    print('media invalid in {}!'.format(csv_file))
    
dfl = []
results_list = []
dfl = [pd.DataFrame(y).sort_values(by=['index1','num_frame']).reset_index(drop=True) for x, y in dfspecies.groupby(by=['sequence_id'], as_index=False)]

results_list = [run_motionmeerkat(df) for df in dfl]

dfspeciesmm = pd.concat(results_list)

if len(dfspecies) > 0:
    dfspecies.drop(['frame_array'],axis=1).to_csv(os.path.join(csv_dir,'{}_species.csv'.format(num_str)), index=False )
if len(dfspeciesmm) > 0:
    dfspeciesmm.to_csv(os.path.join(csv_dir,'{}_species_after_motionm.csv'.format(num_str)), index=False )
if len(dfhumans) > 0:
    dfhumans.to_csv(os.path.join(csv_dir,'{}_humans.csv'.format(num_str)), index=False )
if len(dfmaybe_humans) > 0:
    dfmaybe_humans.to_csv(os.path.join(csv_dir,'{}_maybe_humans.csv'.format(num_str)), index=False )
