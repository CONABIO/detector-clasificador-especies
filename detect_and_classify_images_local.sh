#!/bin/bash

. ~/.camtraproc
NUM_PROCESS=3
ESPECIES_MEX_DIR=$1

#./src/get_videos_to_process.py ${NUM_PROCESSES} ${VIDEOS_JSON_DIR} ${CSV_DIR} ${MODE} ${PL_BATCH_SIZE}

start=$(date +"%s")
# esto es para cada proceso de airflow, o sea que se repite para todos los procesos,
for f in $(seq 1 $(ls $(echo ./${CSV_DIR}/${NUM_PROCESS}/*_files_coor.csv) | wc -l)); do ./src/detect_mammals_images.py "$f"'_files_coor.csv' ${BBOXES_DIR}'/'${NUM_PROCESS} ${CSV_DIR}'/'${NUM_PROCESS} data/models/md_v4.1.0.pb ${ESPECIES_MEX_DIR}; ./src/classify_mammals_images.py ${BBOXES_DIR}'/'${NUM_PROCESS} ${CSV_DIR}'/'${NUM_PROCESS}'/'"$f"'_species.csv' ${RESULTS_DIR}'/'${NUM_PROCESS}'/'"$f"'_results.csv' ${ESPECIES_MEX_DIR} T; rm -rf ${BBOXES_DIR}/${NUM_PROCESS}/*; done
f=1
#./src/classify_mammals_videos.py ${BBOXES_DIR}'/'${NUM_PROCESS} ${CSV_DIR}'/'${NUM_PROCESS}'/'"$f"'_species.csv' ${RESULTS_DIR}'/'${NUM_PROCESS}'/'"$f"'_results.csv'
#./src/test.py
end=$(date +"%s")
date -u -d "0 $end sec - $start sec" +"%H:%M:%S"
