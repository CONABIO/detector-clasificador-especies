import tensorflow as tf
from camtraproc.settings import SIG_INCEPTION_WEIGHTS, TARGET_SIZE, CLASSIFIER_BATCH_SIZE, NCLASSES, BBOXES_DIR
from camtraproc.models import BaseModel, head_for_inception_model, inception_features, get_tuple_list
import numpy as np
import pandas as pd
import os

class FeaturesModel(BaseModel):
    """Antares implementation of Microsoft's Light Boost classifier
    """

    def __init__(self):
        '''
        Example:
            >>> from madmex.modeling.supervised.lgb import Model
            >>> lgb = Model()
            >>> # Write model to db
            >>> lgb.to_db(name='test_model', recipe='mexmad', training_set='no')
            >>> # Read model from db
            >>> lgb2 = Model.from_db('test_model')
        '''
#        head_model = head_for_inception_model(lambda1,NCLASSES,'sigmoid')
#        head_model.load_weights(SIG_INCEPTION_WEIGHTS, by_name=False)
#        head_model.trainable = False

        input_img = tf.keras.Input(shape=(TARGET_SIZE,TARGET_SIZE,3))
        res_img_features =  inception_features(TARGET_SIZE)(input_img)
#        prediction = head_model(res_img_features)

        self.model = tf.keras.Model(inputs=input_img,outputs=res_img_features)

        self.model_name = 'sigmoid_inception'

    #TODO
    def fit(self, X, y):
        X = self.hot_encode_training(X)
        self.model.fit(X,y)

    def predict(self, df, generator, np_dir):
        '''
        Simply passes down the prediction from the underlying model.
        '''
        n_total_val = len(df)
#        index = []
        bb_list = []
        gen = generator.__iter__()
        for f in range(n_total_val//CLASSIFIER_BATCH_SIZE + 1):
            x, bb = next(gen)
            pred = self.model.predict(
                x,
                callbacks=None,
                max_queue_size=1,
                workers=1,
                use_multiprocessing=False,
                verbose=1
            )
#            index.append(ind)

            for g in range(len(pred)):
                path = bb[g][1].split('.')[0] + '.npy'
                np.save(os.path.join(np_dir, path), pred[g])
                if os.path.isfile(os.path.join(np_dir, bb[g][1])):
                    os.remove(os.path.join(np_dir, bb[g][1]))
                else:
                    print("Error: %s file not found" % os.path.join(np_dir, bb[g][1]))

            bb_list.append(bb)

        return bb_list
