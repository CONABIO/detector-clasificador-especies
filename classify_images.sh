#!/bin/bash

. ~/.camtraproc
NUM_PROCESS=2
ESPECIES_MEX_DIR=$1

start=$(date +"%s")
# esto es para cada proceso de airflow, o sea que se repite para todos los procesos. Es el numero de iteracion en un loop
f=1
./src/classify_mammals_images.py ${BBOXES_DIR}'/'${NUM_PROCESS} ${CSV_DIR}'/'${NUM_PROCESS}'/'"$f"'_species.csv' ${RESULTS_DIR}'/'${NUM_PROCESS}'/'"$f"'_results.csv' ${ESPECIES_MEX_DIR} F
#./src/test.py
end=$(date +"%s")
date -u -d "0 $end sec - $start sec" +"%H:%M:%S"
