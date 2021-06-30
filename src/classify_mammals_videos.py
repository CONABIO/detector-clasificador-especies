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

from camtraproc.models.multi_sigmoid_inception import MultiModel
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

if os.path.isfile(bboxes_file):
    df = pd.read_csv(bboxes_file).drop(['index_batch'], axis=1)
    df = df.fillna('0.123|-0.123')
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
                                             x_col=['id','item_file','x','y','w','h'],
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             target_size=TARGET_SIZE,
                                             seed=7233422)

        model = FeaturesModel()
        _ = model.predict(df1, feat_generator, bboxes_dir)

        coor_generator = Features_generator_ex_coo_dist_pot_from_dataframe(dataframe=df1, directory=bboxes_dir,
                                             x_col=['id','item_file','x','y','w','h','prob_ex_coor','dist_pot','score'],
#                                             y_col=,
                                             batch_size=CLASSIFIER_BATCH_SIZE,
                                             seed=7233422)


        model = MultiModel(1e-07)
        all_array = model.predict_taxa(df1, coor_generator,validate=False)

        dff = pd.DataFrame([p[0] for p in all_array], columns=['id'])
        dff['item_file'] = [p[1] for p in all_array]
        dff['frame'] = [int(p[1].split('_')[-2]) for p in all_array]
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
        generator = Features_generator_from_dataframe(dataframe=df2, directory=bboxes_dir,
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
        dff['frame'] = [int(p[1].split('_')[-2]) for p in all_array]
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
