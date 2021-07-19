#!/bin/bash

. ~/.camtraproc
NUM_PROCESS=8
#SHARED_VOLUME=$1

#./src/get_videos_to_process.py ${NUM_PROCESSES} ${VIDEOS_JSON_DIR} ${CSV_DIR} ${MODE} ${PL_BATCH_SIZE}

start=$(date +"%s")
# esto es para cada proceso de airflow, o sea que se repite para todos los procesos,
for f in $(seq 4 $(ls $(echo ./${CSV_DIR}/${NUM_PROCESS}/*_files_coor.csv) | wc -l)); do ./src/detect_mammals_images_videos.py "$f"'_files_coor.csv' ${BBOXES_DIR}'/'${NUM_PROCESS} ${CSV_DIR}'/'${NUM_PROCESS} data/models/md_v4.1.0.pb; done
end=$(date +"%s")
date -u -d "0 $end sec - $start sec" +"%H:%M:%S"