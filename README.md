# detector-clasificador-especies

## Detection with Megadetector and classification with ensemble

Detects fauna and humans in video frames with [Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md). After that we classify 49 species and two genus with an ensemble with one common CNN InceptionResNetV2 headless and five different heads. Some of them uses species occurrences information given coordenates and also potential distribution [*Ceballos, G., S. Blanco, C. González-Salazar, E. Martínez-Meyer. (2006)*](http://www.conabio.gob.mx/informacion/gis/).  

The results of Megadetector are stored in three different csv files:

#_species.csv: Contains Megadetector bounding boxes where fauna is located above a specific threshold
#_humans.csv: Contains Megadetector bounding boxes where humans are located above a specific threshold
#_maybe_humans.csv: Contains Megadetector bounding boxes where humans are located below #_humans.csv threshold and above 0.2

The results of the classifier are stored in RESULTS_DIR

For the implementation of the detector and postprocessing with motion detection inspired by [MotionMeerkat](https://github.com/bw4sz/OpenCV_HummingbirdsMotion) go to branch [MotionMeerkat_postproc](https://github.com/CONABIO/detector-clasificador-especies/tree/MotionMeerkat_postproc).  

## Deployment

### 1. Clone thios repo
```
git clone https://github.com/CONABIO/detector-clasificador-especies
```

### 2. Install
```
pip install ./detector-clasificador-especies --no-deps
```

### 3. Set variables file in home
Create the file `.camtraproc` in your home directory with the following information:
```
NUM_PROCESSES=10
PL_BATCH_SIZE=10
VIDEOS_JSON_DIR=data/tmp
CSV_DIR=data/csv
RESULTS_DIR=data/results
BBOXES_DIR=data/tmp
ITEM_TYPE=147
ITEM_LIST_ENDPOINT=http://irekuaapi-env.eba-gj4jy7ue.us-west-2.elasticbeanstalk.com/api/collections/v1/collection_items/?item_type=147 #146 img, 147 video
ITEM_DETAIL_ENDPOINT=
MODE=irekua #s3,irekua,manual
USER=<irekua-username>
PSSWD=<irekua-password>
AWS_SERVER_PUBLIC_KEY=<aws-public-key>
AWS_SERVER_SECRET_KEY=<AWS-secret-key>
DETECTOR_THRESHOLD=0.7
DETECTOR_BATCH_SIZE=16
CATEGORY_FILE=data/resources/csv/category_id_fam_gen_species_names_20201203.csv
TARGET_SIZE=299
CLASSIFIER_BATCH_SIZE=32
SIG_INCEPTION_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_sigmoid_num_1000/weights.231-3.29-0.50.hdf5
MODEL_17_2_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_17-2/weights.223-3.31-0.69.hdf5
MODEL_17_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_17/weights.235-3.33-0.67.hdf5
MODEL_13_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_13/weights.380-3.42-0.58.hdf5
MODEL_13_2_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_13-2/weights.524-3.42-0.56.hdf5
NCLASSES=51
EX_COOR_FILENAME=data/resources/pkls/coord_all_uniq_greater_15_species_ex_coor_sigmoid_str_coo.pkl
DIST_POT_FILENAME=data/resources/pkls/coord_all_uniq_greater_15_species_dist_pot_sigmoid_str_coo.pkl
THRESH1=0.97
THRESH2=0.94
THRESH3=0.6
```

### 4. Run detection and classification
```
./detect_and_classify_videos.sh
```

**Note:**
you could use the dockerfile attached for deployment
