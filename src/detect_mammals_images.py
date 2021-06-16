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
from camtraproc.detection.detection import generate_detections, get_images, load_model
import pandas as pd
import os
import sys


csv_file = sys.argv[1]
bboxes_dir = sys.argv[2]
csv_dir = sys.argv[3]
detector_file = sys.argv[4]
dirpath = sys.argv[5]
num_str = csv_file.split('_')[0]
species = []
humans = []
maybe_humans = []

df = pd.read_csv(os.path.join(csv_dir,csv_file))

detection_graph4 = load_model(detector_file)

for i in range(len(df)):
    images,ndf = get_images(dirpath,df.iloc[i]['item_file'],df.iloc[i])
    if images is not False:
        species_df, humans_df, maybe_humans_df = generate_detections(detection_graph4,images,bboxes_dir,ndf)
        species.append(species_df)
        humans.append(humans_df)
        maybe_humans.append(maybe_humans_df)

dfspecies = pd.concat(species)
dfhumans = pd.concat(humans)
dfmaybe_humans = pd.concat(maybe_humans)

if len(dfspecies) > 0:
    dfspecies.to_csv(os.path.join(csv_dir,'{}_species.csv'.format(num_str)), index=False )
if len(dfhumans) > 0:
    dfhumans.to_csv(os.path.join(csv_dir,'{}_humans.csv'.format(num_str)), index=False )
if len(dfmaybe_humans) > 0:
    dfmaybe_humans.to_csv(os.path.join(csv_dir,'{}_maybe_humans.csv'.format(num_str)), index=False )
