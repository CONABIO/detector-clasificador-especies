#!/usr/bin/env python3

from camtraproc.settings import CATEGORY_FILE, EX_COOR_FILENAME, DIST_POT_FILENAME, CLASSIFIER_BATCH_SIZE, TARGET_SIZE, EX_COOR_FILENAME, DIST_POT_FILENAME, CATEGORY_FILE
from camtraproc.models.multi_sigmoid_inception import MultiModel
from camtraproc.models.sigmoid_inception import SimpleModel
from camtraproc.models.inception_features import FeaturesModel
from camtraproc.classification.generator import Features_generator_for_feat_from_dataframe, Features_generator_ex_coo_dist_pot_from_dataframe, Features_generator_from_dataframe
from camtraproc.detection.detection import crop_generator
import PIL.Image
import numpy as np
import pandas as pd
import cv2
import os
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

bboxes_dir = sys.argv[1]
bboxes_file = sys.argv[2]
results_file = sys.argv[3]
directory = sys.argv[4]
from_scratch = sys.argv[5]

df = pd.read_csv(bboxes_file).drop(['index_batch'], axis=1)
dfca = pd.read_csv(CATEGORY_FILE)
dfec = pd.read_pickle(EX_COOR_FILENAME)
df = df.merge(dfec, on=['latlong'],how='left')
dfdp = pd.read_pickle(DIST_POT_FILENAME)
df = df.merge(dfdp, on=['latlong'],how='left')


df1 = df[~df.isnull().any(axis=1)].drop(['latlong'],axis=1)
dft = df[df['prob_ex_coor'].isnull()].drop(['latlong'],axis=1)
df2 = dft[dft['dist_pot'].isnull()]

if from_scratch == 'F':
    df1bb = df1.copy().drop(['item_file'], axis=1)
    df2bb = df2.copy().drop(['item_file'], axis=1)
    df1bb['item_file'] = df1.apply(lambda x: x.item_file.split('.')[0] + '_bb0.' + x.item_file.split('.')[1], axis=1 )
    df2bb['item_file'] = df2.apply(lambda x: x.item_file.split('.')[0] + '_bb0.' + x.item_file.split('.')[1], axis=1 )
else:
    df1bb = df1
    df2bb = df2

if len(df1) > 0:
    if from_scratch == 'F':
        for r in np.array(df1):
           # image = PIL.Image.open(os.path.join(directory, r[0])).convert("RGB"); image = np.array(image)# 
            image = cv2.imread(os.path.join(directory, r[0]))
            x = int(r[1]*image.shape[1] + 0.5)
            y = int(r[2]*image.shape[0] + 0.5)
            w = int(r[3]*image.shape[1] + 0.5)
            h = int(r[4]*image.shape[0] + 0.5)
           # image_to_write = cv2.cvtColor(crop_generator(image,x,y,w,h), cv2.COLOR_RGB2BGR)
            image_to_write = crop_generator(image,x,y,w,h)
            os.makedirs(os.path.dirname(os.path.join(bboxes_dir, r[0])), exist_ok=True)
            cv2.imwrite(os.path.join(bboxes_dir, r[0].split('.')[0] + '_bb0.' + r[0].split('.')[1]), image_to_write)

    feat_generator = Features_generator_for_feat_from_dataframe(dataframe=df1bb, directory=bboxes_dir,
                                             x_col=['id','item_file','x','y','w','h'],
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             target_size=TARGET_SIZE,
                                             seed=7233422)

    model = FeaturesModel()
    _ = model.predict(df1, feat_generator, bboxes_dir)

    coor_generator = Features_generator_ex_coo_dist_pot_from_dataframe(dataframe=df1bb, directory=bboxes_dir,
                                             x_col=['id','item_file','x','y','w','h','prob_ex_coor','dist_pot','score'],
#                                             y_col=,
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             seed=7233422)


    model = MultiModel(1e-07)
    all_array = model.predict_taxa(df1, coor_generator,validate=False)

    dff = pd.DataFrame([p[0] for p in all_array], columns=['id'])
    dff['item_file'] = [p[1] for p in all_array]
#    dff['frame'] = [int(p[1].split('_')[-2]) for p in all_array]
    dff['x'] = [p[2] for p in all_array]
    dff['y'] = [p[3] for p in all_array]
    dff['w'] = [p[4] for p in all_array]
    dff['h'] = [p[5] for p in all_array]
    dff['bboxes_score'] = [p[8] for p in all_array]
    dff['pred1'] = [dfca[dfca['category_id'] == p[9]][p[11] + '_name'].values[0] if p[9] is not None else None for p in all_array]
    dff['score1'] = [p[10] for p in all_array]
    dff['taxa_level1'] = [p[11] for p in all_array]
    dff['pred2'] = [dfca[dfca['category_id'] == p[12]][p[14] + '_name'].values[0]  if p[12] is not None else None for p in all_array]
    dff['score2'] = [p[13] for p in all_array]
    dff['taxa_level2'] = [p[14] for p in all_array]
    dff['pred3'] = [dfca[dfca['category_id'] == p[15]][p[17] + '_name'].values[0] if p[15] is not None else None for p in all_array]
    dff['score3'] = [p[16] for p in all_array]
    dff['taxa_level3'] = [p[17] for p in all_array]
    dff['pred4'] = [dfca[dfca['category_id'] == p[18]][p[20] + '_name'].values[0] if p[18] is not None else None for p in all_array]
    dff['score4'] = [p[19] for p in all_array]
    dff['taxa_level4'] = [p[20] for p in all_array]
    dff['pred5'] = [dfca[dfca['category_id'] == p[21]][p[23] + '_name'].values[0] if p[21] is not None else None for p in all_array]
    dff['score5'] = [p[22] for p in all_array]
    dff['taxa_level5'] = [p[23] for p in all_array]


    if len(dff) > 0:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        dff.to_csv(results_file, index=False)

if len(df2) > 0:
    if from_scratch == 'F':
        for r in np.array(df2):
            image = cv2.imread(os.path.join(directory, r[0]))
            x = int(r[1]*image.shape[1] + 0.5)
            y = int(r[2]*image.shape[0] + 0.5)
            w = int(r[3]*image.shape[1] + 0.5)
            h = int(r[4]*image.shape[0] + 0.5)
            #image_to_write = cv2.cvtColor(crop_generator(image,x,y,w,h), cv2.COLOR_RGB2BGR)
            image_to_write = crop_generator(image,x,y,w,h)
            os.makedirs(os.path.dirname(os.path.join(bboxes_dir, r[0])), exist_ok=True)
            cv2.imwrite(os.path.join(bboxes_dir, r[0].split('.')[0] + '_bb0.' + r[0].split('.')[1]), image_to_write)

    generator = Features_generator_from_dataframe(dataframe=df2bb, directory=bboxes_dir,
                                             x_col=['id','item_file','x','y','w','h','score'],
#                                             y_col=['category_id'],
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             target_size=TARGET_SIZE,
                                             seed=7233422)

    model = SimpleModel(1e-07)
    all_array = None
    all_array = model.predict_taxa(df2, generator, validate=False, bayesian=False)

    dff = None
    dff = pd.DataFrame([p[0] for p in all_array], columns=['id'])
    dff['item_file'] = [p[1] for p in all_array]
#    dff['frame'] = [int(p[1].split('_')[-2]) for p in all_array]
    dff['x'] = [p[2] for p in all_array]
    dff['y'] = [p[3] for p in all_array]
    dff['w'] = [p[4] for p in all_array]
    dff['h'] = [p[5] for p in all_array]
    dff['bboxes_score'] = [p[6] for p in all_array]
    dff['pred'] = [dfca[dfca['category_id'] == p[7]][p[9] + '_name'].values[0] if p[7] is not None else None for p in all_array]
    dff['score'] = [p[8] for p in all_array]
    dff['taxa_level'] = [p[9] for p in all_array]

    if len(dff) > 0:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        dff.to_csv(results_file.split('.')[0] + '_no_coor.csv', index=False)
