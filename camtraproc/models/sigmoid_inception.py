import tensorflow as tf
from camtraproc.settings import SIG_INCEPTION_WEIGHTS, TARGET_SIZE, CLASSIFIER_BATCH_SIZE, NCLASSES, THRESH1, THRESH2, THRESH3
from camtraproc.models import BaseModel, head_for_inception_model, inception_features, get_tuple_list
import numpy as np

class SimpleModel(BaseModel):
    """Antares implementation of Microsoft's Light Boost classifier
    """

    def __init__(self, lambda1=1e-07):
        '''
        Example:
            >>> from madmex.modeling.supervised.lgb import Model
            >>> lgb = Model()
            >>> # Write model to db
            >>> lgb.to_db(name='test_model', recipe='mexmad', training_set='no')
            >>> # Read model from db
            >>> lgb2 = Model.from_db('test_model')
        '''
        head_model = head_for_inception_model(lambda1,NCLASSES,'sigmoid')
        head_model.load_weights(SIG_INCEPTION_WEIGHTS, by_name=False)
        head_model.trainable = False

        input_img = tf.keras.Input(shape=(TARGET_SIZE,TARGET_SIZE,3))
        res_img_features =  inception_features(TARGET_SIZE)(input_img)
        prediction = head_model(res_img_features)

        self.model = tf.keras.Model(inputs=input_img,outputs=prediction)

        self.model_name = 'sigmoid_inception'

    #TODO
    def fit(self, X, y):
        X = self.hot_encode_training(X)
        self.model.fit(X,y)

    def predict(self, df, generator, validate=False, bayesian=False):
        '''
        Simply passes down the prediction from the underlying model.
        '''
        n_total_val = len(df)
        label = []
        predict_list = []
        bb_list = []
        gen = generator.__iter__()
        for f in range(n_total_val//CLASSIFIER_BATCH_SIZE + 1):
            if validate:
                x, y, bb = next(gen)
                pred = self.model.predict(
                    x, 
                    callbacks=None, 
                    max_queue_size=1, 
                    workers=1,
                    use_multiprocessing=False, 
                    verbose=1
                )
                label.append(y)
                label_list = [p for f in label for p in f]
                predict_list.append(pred)
                bb_list.append(bb)
            else:
                x, bb = next(gen)
                pred = self.model.predict(
                    x, 
                    callbacks=None, 
                    max_queue_size=1, 
                    workers=1,
                    use_multiprocessing=False, 
                    verbose=1
                )
                predict_list.append(pred)
                bb_list.append(bb)

        predict_li = [p for f in predict_list for p in f]
        bb_array = np.concatenate(bb_list, axis=0)

        if bayesian:
            pred_list, conf_list, taxa_list = get_tuple_list([predict_list[f][i][0]*bb_list[f][i][6]
                                                 for f in range(len(predict_list)) for i in range(len(predict_list[f]))])
        else:
            pred_list, conf_list, taxa_list = get_tuple_list([predict_list[f][i][0]
                                                 for f in range(len(predict_list)) for i in range(len(predict_list[f]))])

        all_array = np.concatenate([bb_array,np.array(pred_list)[:,None],np.array(conf_list)[:,None],np.array(taxa_list)[:,None]], axis=1)

        if validate:
            return all_array, label_list
        else:
            return all_array



    def predict_taxa(self, df, generator, validate=False, bayesian=False):
        n_total_val = len(df)
        label = []
        predict_list = []
        bb_list = []
        gen = generator.__iter__()
        for f in range(n_total_val//CLASSIFIER_BATCH_SIZE + 1):
            if validate:
                x, y, bb = next(gen)
                pred = self.model.predict(
                    x, 
                    callbacks=None, 
                    max_queue_size=1, 
                    workers=1,
                    use_multiprocessing=False, 
                    verbose=1
                )
                label.append(y)
                label_list = [p for f in label for p in f]
                predict_list.append(pred)
                bb_list.append(bb)
            else:
                x, bb = next(gen)
                pred = self.model.predict(
                    x, 
                    callbacks=None, 
                    max_queue_size=1, 
                    workers=1,
                    use_multiprocessing=False, 
                    verbose=1
                )
                predict_list.append(pred)
                bb_list.append(bb)

        predict_li = [p for f in predict_list for p in f]
        bb_array = np.concatenate(bb_list, axis=0)

        if bayesian:
            pred_list, conf_list, taxa_list = get_tuple_list([predict_list[f][i]*bb_list[f][i][6]
                                                 for f in range(len(predict_list)) for i in range(len(predict_list[f]))],
                                                 THRESH1,THRESH2,THRESH3)
        else:
            pred_list, conf_list, taxa_list = get_tuple_list([predict_list[f][i]
                                                 for f in range(len(predict_list)) for i in range(len(predict_list[f]))],
                                                 THRESH1,THRESH2,THRESH3)

        all_array = np.concatenate([bb_array,np.array(pred_list)[:,None],np.array(conf_list)[:,None],np.array(taxa_list)[:,None]], axis=1)

        if validate:
            return all_array, label_list
        else:
            return all_array
