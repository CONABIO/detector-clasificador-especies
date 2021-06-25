from dotenv.main import load_dotenv
import warnings
import os

dotenv_path = os.path.join(os.path.expanduser('~'), '.camtraproc')

if os.path.isfile(dotenv_path):
    load_dotenv(dotenv_path)
else:
    warnings.warn('No configuration file found in %s' % dotenv_path)

NUM_PROCESSES=int(os.getenv('NUM_PROCESSES'))
PL_BATCH_SIZE=int(os.getenv('PL_BATCH_SIZE'))
VIDEOS_JSON_DIR=os.getenv('VIDEOS_JSON_DIR')
CSV_DIR=os.getenv('CSV_DIR')
RESULTS_DIR=os.getenv('RESULTS_DIR')
BBOXES_DIR=os.getenv('BBOXES_DIR')
ITEM_TYPE=os.getenv('ITEM_TYPE')
ITEM_LIST_ENDPOINT=os.getenv('ITEM_LIST_ENDPOINT')
ITEM_DETAIL_ENDPOINT=os.getenv('ITEM_DETAIL_ENDPOINT')
MODE=os.getenv('MODE')
USER=os.getenv('USER')
PSSWD=os.getenv('PSSWD')
AWS_SERVER_PUBLIC_KEY=os.getenv('AWS_SERVER_PUBLIC_KEY')
AWS_SERVER_SECRET_KEY=os.getenv('AWS_SERVER_SECRET_KEY')
DETECTOR_THRESHOLD=float(os.getenv('DETECTOR_THRESHOLD'))
DETECTOR_BATCH_SIZE=int(os.getenv('DETECTOR_BATCH_SIZE'))
CATEGORY_FILE=os.getenv('CATEGORY_FILE')
TARGET_SIZE=int(os.getenv('TARGET_SIZE'))
CLASSIFIER_BATCH_SIZE=int(os.getenv('CLASSIFIER_BATCH_SIZE'))
SIG_INCEPTION_WEIGHTS=os.getenv('SIG_INCEPTION_WEIGHTS')
MODEL_17_2_WEIGHTS=os.getenv('MODEL_17_2_WEIGHTS')
MODEL_17_WEIGHTS=os.getenv('MODEL_17_WEIGHTS')
MODEL_13_WEIGHTS=os.getenv('MODEL_13_WEIGHTS')
MODEL_13_2_WEIGHTS=os.getenv('MODEL_13_2_WEIGHTS')
NCLASSES=int(os.getenv('NCLASSES'))
EX_COOR_FILENAME=os.getenv('EX_COOR_FILENAME')
DIST_POT_FILENAME=os.getenv('DIST_POT_FILENAME')
THRESH1=float(os.getenv('THRESH1'))
THRESH2=float(os.getenv('THRESH2'))
THRESH3=float(os.getenv('THRESH3'))
