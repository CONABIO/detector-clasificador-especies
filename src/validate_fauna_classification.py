#!/usr/bin/env python3
"""

input:
${BBOXES_DIR}'/'${NUM_PROCESS}
${CSV_DIR}'/'${NUM_PROCESS}'/'"$f"'_species.csv'
${RESULTS_DIR}'/'${NUM_PROCESS}'/'"$f"'_results.csv'
    bboxes_dir
    file
    results_file
"""

from camtraproc.models.multi_sigmoid_inception import Model
from camtraproc.models.sigmoid_inception import SimpleModel
from camtraproc.models.inception_features import FeaturesModel
from camtraproc.classification.generator import Features_generator_for_feat_from_dataframe, Features_generator_ex_coo_dist_pot_from_dataframe, Features_generator_from_dataframe
from camtraproc.settings import CLASSIFIER_BATCH_SIZE, TARGET_SIZE, EX_COOR_FILENAME, DIST_POT_FILENAME, CATEGORY_FILE
import pandas as pd
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

bboxes_dir = sys.argv[1]
bboxes_file = sys.argv[2]
results_file = sys.argv[3]

df = pd.read_csv(bboxes_file).drop(['index_batch'], axis=1)
dfca = pd.read_csv(CATEGORY_FILE)
dfec = pd.read_pickle(EX_COOR_FILENAME)
df = df.merge(dfec, on=['latlong'],how='left')
dfdp = pd.read_pickle(DIST_POT_FILENAME)
df = df.merge(dfdp, on=['latlong'],how='left')


df1 = df[~df.isnull().any(axis=1)].drop(['latlong'],axis=1)
dft = df[df['prob_ex_coor'].isnull()].drop(['latlong'],axis=1)
df2 = dft[dft['dist_pot'].isnull()]


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
#    index, bb_list, pred_list, conf_list, taxa_list, pred_bayes_list, conf_bayes_list, taxa_bayes_list = model.predict_taxa(df1, coor_generator)
    index_list, all_array = model.predict_taxa(df1, coor_generator)

    dff = pd.DataFrame([p for f in index_list for p in f], columns=['index'])
    dff['filename'] = [p[0] for p in all_array]
    dff['frame'] = [int(p[0].split('_')[-2]) for p in all_array]
    dff['bboxes'] = [p[1:5] for p in all_array]
    dff['pred1'] = [dfca[dfca['category_id'] == p[7]][p[9] + '_name'].values[0] if p[7] is not None else None for p in all_array]
    dff['confidence1'] = [p[8] for p in all_array]
    dff['taxa_level1'] = [p[9] for p in all_array]
    dff['pred2'] = [dfca[dfca['category_id'] == p[10]][p[12] + '_name'].values[0]  if p[10] is not None else None for p in all_array]
    dff['confidence2'] = [p[11] for p in all_array]
    dff['taxa_level2'] = [p[12] for p in all_array]
    dff['pred3'] = [dfca[dfca['category_id'] == p[13]][p[15] + '_name'].values[0] if p[13] is not None else None for p in all_array]
    dff['confidence3'] = [p[14] for p in all_array]
    dff['taxa_level3'] = [p[15] for p in all_array]
    dff['pred4'] = [dfca[dfca['category_id'] == p[16]][p[18] + '_name'].values[0] if p[16] is not None else None for p in all_array]
    dff['confidence4'] = [p[17] for p in all_array]
    dff['taxa_level4'] = [p[18] for p in all_array]
    dff['pred1'] = [dfca[dfca['category_id'] == p[19]][p[21] + '_name'].values[0] if p[19] is not None else None for p in all_array]
    dff['confidence1'] = [p[20] for p in all_array]
    dff['taxa_level1'] = [p[21] for p in all_array]


    if len(dff) > 0:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        dff.to_csv(results_file, index=False)

if len(df2) > 0:
    generator = Features_generator_from_dataframe(dataframe=df2, directory=bboxes_dir,
                                             x_col=['item_file','x','y','w','h'],
                                             y_col=['id','prob_ex_coor'],
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             target_size=TARGET_SIZE,
                                             seed=7233422)

    model = SimpleModel(1e-07)
    index, bb_list, pred_list, conf_list, taxa_list = model.predict_taxa(df2, generator)

    dff = None
    dff = pd.DataFrame([p for f in index for p in f], columns=['index'])
    dff['filename'] = [p[0] for f in bb_list for p in f]
    dff['frame'] = [int(p[0].split('_')[-2]) for f in bb_list for p in f]
    dff['bboxes'] = [p[1:] for f in bb_list for p in f]
    dff['pred1'] = pred_list
    dff['confidence1'] = conf_list
    dff['taxa_level1'] = taxa_list

    if len(dff) > 0:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        dff.to_csv(results_file.split('.')[0] + '_no_coor.csv', index=False)


