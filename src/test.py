#!/usr/bin/env python3

from camtraproc.models.multi_sigmoid_inception import Model
from camtraproc.models.sigmoid_inception import SimpleModel
from camtraproc.models.inception_features import FeaturesModel
from camtraproc.classification.generator import Features_generator_ex_coo_dist_pot_from_dataframe, Features_generator_for_feat_from_dataframe, Features_generator_from_dataframe
from camtraproc.settings import CLASSIFIER_BATCH_SIZE, TARGET_SIZE, EX_COOR_FILENAME, DIST_POT_FILENAME, BBOXES_DIR, CSV_DIR, RESULTS_DIR 
import pandas as pd
import sys
import os

#path='/home/common_user/workspace_c/repos/species_models'
#os.chdir(path)
bboxes_dir = os.path.join(BBOXES_DIR,str(1))
bboxes_file = os.path.join(CSV_DIR,str(1),'1_species.csv')
results_file = os.path.join(RESULTS_DIR,str(1),'1_results.csv')

df = pd.read_csv(bboxes_file).drop(['index_batch'], axis=1)
dfec = pd.read_pickle(EX_COOR_FILENAME)
df = df.merge(dfec, on=['latlong'],how='left')
dfdp = pd.read_pickle(DIST_POT_FILENAME)
df = df.merge(dfdp, on=['latlong'],how='left')


df1 = df[~df.isnull().any(axis=1)].drop(['latlong'],axis=1)
dft = df[df['prob_ex_coor'].isnull()].drop(['latlong'],axis=1)
df2 = dft[dft['dist_pot'].isnull()]

print(df1)

if len(df1) > 0:
    feat_generator = Features_generator_for_feat_from_dataframe(dataframe=df1, directory=bboxes_dir,
                                             x_col=['item_file','x','y','w','h'],
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             target_size=TARGET_SIZE,
                                             seed=7233422)

    model = FeaturesModel()
    _ = model.predict(df1, feat_generator, bboxes_dir)

    coor_generator = Features_generator_ex_coo_dist_pot_from_dataframe(dataframe=df1, directory=bboxes_dir,
                                             x_col=['item_file','x','y','w','h','prob_ex_coor','dist_pot'],
                                             y_col=['id'],
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             seed=7233422)
    model = Model(1e-07)
#    index, bb_list, pred_list, score_list, taxa_list, pred_bayes_list, score_bayes_list, taxa_bayes_list = model.predict_taxa(df1, coor_generator)
    index_list, all_array = model.predict_taxa(df1, coor_generator)
    print(all_array.shape)
else:
    print('algo esta mal')
