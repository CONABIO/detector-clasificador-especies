# detector-clasificador-especies

## Detection with Megadetector and postprocessing with MotionMeerkat

Detects fauna and humans in images and video frames with [Megadetector](https://github.com/microsoft/CameraTraps/blob/master/megadetector.md). After that we postprocess the results with an algorithm derived from [MotionMeerkat](https://github.com/bw4sz/OpenCV_HummingbirdsMotion). We group images and videos taken maximum SEQ_TIME_DELTA seconds apart into sequences and use video frames to learn the background of the scene in order to detect motion in images and video frames of the same sequence. This procedure can be done using both MotionMeerkat methods (MOG and Acc) and also in two different forms: using the whole images and frames or a roi.  

In case of roi method, we considered that images with bounding boxes greater than 0.15x0.15 (relative size) and with scores greater than 0.985 have fauna, no matter that no motion was detected. This values can be changed [here](https://github.com/CONABIO/detector-clasificador-especies/blob/MotionMeerkat_postproc/camtraproc/detection/motionm_bbox.py#L175)  

The results are stored in four different csv files:  

#_species.csv: Contains Megadetector bounding boxes where fauna is located above a specific threshold  
#_humans.csv: Contains Megadetector bounding boxes where humans are located above a specific threshold  
#_maybe_humans.csv: Contains Megadetector bounding boxes where humans are located below #_humans.csv threshold and above 0.2  
#_species_after_motionm.csv: Contains results after motion detection  

After postprocessing, #_species_after_motionm.csv is written into irekua as predictions and the csv is deleted. In order to skip this step, delete `./src/post_predictions.py` [here](https://github.com/CONABIO/detector-clasificador-especies/blob/MotionMeerkat_postproc/detect_images_videos_motion_bbox.sh#L11) or [here](https://github.com/CONABIO/detector-clasificador-especies/blob/MotionMeerkat_postproc/detect_images_videos_motion_all_image.sh#L11).  

## Deployment

### 1. Clone thios repo
```
git clone https://github.com/CONABIO/detector-clasificador-especies
```

### 2. Install
```
cd detector-clasificador-especies
cd checkout MotionMeerkat_postproc
pip install ./detector-clasificador-especies --no-deps
```

### 3. Set variables file in home
Create the file `.camtraproc` in your home directory with the following information:
```
NUM_PROCESSES=10 #Number of processes. Usefull to divide the dataset files in different directories 
PL_BATCH_SIZE=5 #Number of sequences in each batch file inside the processes directories (CSV_DIR/#)
VIDEOS_JSON_DIR=data/tmp # temp directory
CSV_DIR=data/csv # directory where batch files and species, humans and maybe_humans files are stored
RESULTS_DIR=data/results # results directory for classifier.
BBOXES_DIR=data/tmp # bounding boxes directory
ITEM_TYPE= # item type in irekua
ITEM_LIST_ENDPOINT=http://irekuaapi-env.eba-gj4jy7ue.us-west-2.elasticbeanstalk.com/api/collections/v1/collection_items/ # list of media files in irekua's api
ITEM_DETAIL_ENDPOINT=
MRUN_LIST_ENDPOINT=http://irekuaapi-env.eba-gj4jy7ue.us-west-2.elasticbeanstalk.com/api/models/v1/model_runs/
MPRED_ENDPOINT=http://irekuaapi-env.eba-gj4jy7ue.us-west-2.elasticbeanstalk.com/api/models/v1/model_predictions/ # list of predictions in irekua's api
MODE=irekua #s3,irekua,manual -> media files read can be done from irekua, from s3 or in case of manual, metadata must be inside `data/tmp`
USER=<irekua's username>
PSSWD=<irekua's password>
AWS_SERVER_PUBLIC_KEY=<aws-public-key>
AWS_SERVER_SECRET_KEY=<AWS-secret-key>
DETECTOR_THRESHOLD=0.7 # bounding boxes of species above this threshold are considered
DETECTOR_BATCH_SIZE=2
CATEGORY_FILE=data/resources/csv/category_id_fam_gen_species_names_20201203.csv # species names and id's for classifier
TARGET_SIZE=299 # size of images for classifier
CLASSIFIER_BATCH_SIZE=32
SIG_INCEPTION_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_sigmoid_num_1000/weights.231-3.29-0.50.hdf5 # model's weights for classifier
MODEL_17_2_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_17-2/weights.223-3.31-0.69.hdf5 # model's weights for classifier
MODEL_17_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_17/weights.235-3.33-0.67.hdf5 # model's weights for classifier
MODEL_13_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_13/weights.380-3.42-0.58.hdf5 # model's weights for classifier
MODEL_13_2_WEIGHTS=data/resources/weights/inceptionRNv2_ap_3_mod_1_w_coo_w_weights_softmax_num_2_3_13-2/weights.524-3.42-0.56.hdf5 # model's weights for classifier
NCLASSES=51 # number of classes of classifier
EX_COOR_FILENAME=data/resources/pkls/coord_all_uniq_greater_15_species_ex_coor_sigmoid_str_coo.pkl # species occurrance info in specific coordinates
DIST_POT_FILENAME=data/resources/pkls/coord_all_uniq_greater_15_species_dist_pot_sigmoid_str_coo.pkl # species potential distribution in specific coordinates
THRESH1=0.97 # classifier threshold for species
THRESH2=0.94 # classifier threshold for genus
THRESH3=0.6 # classifier threshold for family
WITH_MOTION_SEQ=yes # leave empty if motionmeerkat postprocessing will not be performed
IMAGE_TYPE=146 # image type id in irekua
VIDEO_TYPE=147 # video type id in irekua
SEQ_TIME_DELTA=20 # maximum seconds between media in a sequence
SUBMETHOD=MOG #MOG,Acc # Method to determine background in images
ACCAVG=0.35
THRESHT=30
MOGVAR=25
MOGLEARNING=0.14 
MINSIZE=20
MODEL_VERSION=1 # model version id in irekua
MODEL_RUN=1 # model run id in irekua
ANNOT_TYPE=1 # annotation type id in irekua
EVENT_TYPE_FAUNA=1 # detector's fauna label id in irekua
EVENT_TYPE_ANTROP=2 # detector's humans or human's activity label id in irekua
LABEL=2 # any other label id to add in irekua i.e. Animalia
```

### 4. Choose postprocessing method and run (whole image or roi)

- for whole image:  
```
./detect_images_videos_motion_all_image.sh
```
- for roi:
```
./detect_images_videos_motion_bbox.sh
```

Inside those bash scripts specify the number of process you will execute (one of the NUM_PROCESSES) in line 4.  

**Note:** you could use the dockerfile attached for deployment