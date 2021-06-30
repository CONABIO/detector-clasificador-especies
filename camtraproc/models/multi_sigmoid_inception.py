import tensorflow as tf
from camtraproc.settings import SIG_INCEPTION_WEIGHTS, MODEL_17_2_WEIGHTS, MODEL_17_WEIGHTS, MODEL_13_WEIGHTS,  MODEL_13_2_WEIGHTS, TARGET_SIZE, CLASSIFIER_BATCH_SIZE, NCLASSES
from camtraproc.models import BaseModel, head_for_inception_model, inception_features, get_tuple_list, dense_coo_layer
import numpy as np

class MultiModel(BaseModel):
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
#        input_img = tf.keras.Input(shape=(TARGET_SIZE,TARGET_SIZE,3))
#        res_img_features =  inception_features(TARGET_SIZE)(input_img)
#        model1 = tf.keras.Model(inputs=input_img, outputs=res_img_features)
#        model1.summary()

        input_inception = tf.keras.Input(shape=(8,8,1536))
        head_model = head_for_inception_model(lambda1,NCLASSES,'sigmoid')
        head_model.load_weights(SIG_INCEPTION_WEIGHTS, by_name=False)
        head_model.trainable = False

#        input_img = tf.keras.Input(shape=(TARGET_SIZE,TARGET_SIZE,3))
#        res_img_features =  inception_features(TARGET_SIZE)(input_img)
        img_features = head_model(input_inception)

        input_ex_coo = tf.keras.Input(shape=(51,))
        sum_lambda = tf.keras.layers.Lambda(lambda tensors:tf.keras.backend.abs(tensors[0] + tensors[1]))

        sum_layer17_2 = sum_lambda([img_features, input_ex_coo])
        net17_2 = tf.keras.layers.Dense(51,activation='softmax',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-07))(sum_layer17_2)
        prediction17_2 = tf.keras.Model(inputs=[input_inception,input_ex_coo],outputs=net17_2)
        prediction17_2.load_weights(MODEL_17_2_WEIGHTS, by_name=False)
        prediction17_2.trainable = False

        prediction17_2_layer = prediction17_2([input_inception,input_ex_coo])

        coo_layer17 =  dense_coo_layer(input_ex_coo,lambda1,'sigmoid')
        sum_layer17 = sum_lambda([img_features, coo_layer17])
        net17 = tf.keras.layers.Dense(51,activation='softmax',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-07))(sum_layer17)
        prediction17 = tf.keras.Model(inputs=[input_inception,input_ex_coo],outputs=net17)
        prediction17.load_weights(MODEL_17_WEIGHTS, by_name=False)
        prediction17.trainable = False

        prediction17_layer = prediction17([input_inception,input_ex_coo])

        input_dist_pot = tf.keras.Input(shape=(51,))

        coo_layer13 =  dense_coo_layer(input_dist_pot,lambda1,'sigmoid')
        sum_layer13 = sum_lambda([img_features, coo_layer13])
        net13 = tf.keras.layers.Dense(51,activation='softmax',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-07))(sum_layer13)
        prediction13 = tf.keras.Model(inputs=[input_inception,input_dist_pot],outputs=net13)
        prediction13.load_weights(MODEL_13_WEIGHTS, by_name=False)
        prediction13.trainable = False

        prediction13_layer = prediction13([input_inception,input_dist_pot])

        sum_layer13_2 = sum_lambda([img_features, input_dist_pot])
        net13_2 = tf.keras.layers.Dense(51,activation='softmax',
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-07))(sum_layer13_2)
        prediction13_2 = tf.keras.Model(inputs=[input_inception,input_dist_pot],outputs=net13_2)
        prediction13_2.load_weights(MODEL_13_2_WEIGHTS, by_name=False)
        prediction13_2.trainable = False

        prediction13_2_layer = prediction13_2([input_inception,input_dist_pot])

        con_lambda = tf.keras.layers.Lambda(lambda tensors:tf.keras.backend.stack([tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]],axis=1))
        prediction = con_lambda([img_features, prediction17_2_layer, prediction17_layer, prediction13_layer, prediction13_2_layer])

        self.model = tf.keras.Model(inputs=[input_inception,input_ex_coo,input_dist_pot],outputs=prediction)

        self.model_name = 'multi_sigmoid_inception'

    #TODO
    def fit(self, X, y):
        X = self.hot_encode_training(X)
        self.model.fit(X,y)
    #TODO
    def predict(self, df, generator):
        '''
        Simply passes down the prediction from the underlying model.
        '''
        n_total_val = len(df)
        index = []
        predict_list = []
        predict_bayes_list = []
        bb_list = []
        gen = generator.__iter__()
        for f in range(n_total_val//CLASSIFIER_BATCH_SIZE + 1):
            x, ind, coordenates, bb = next(gen)
            pred = self.model.predict(
                x,
                callbacks=None,
                max_queue_size=1,
                workers=1,
                use_multiprocessing=False,
                verbose=1
            )
            index.append(ind)
            predict_list.append(pred)
            predict_bayes_list.append(pred*coordenates)
            bb_list.append(bb)

        return index, bb_list, [p for f in predict_list for p in f], [p for f in predict_bayes_list for p in f]


    def predict_taxa(self, df, generator,validate=False):
        n_total_val = len(df)
#        index = []
        predict_list = []
#        predict_bayes_list = []
        bb_list = []
        gen = generator.__iter__()
        for f in range(n_total_val//CLASSIFIER_BATCH_SIZE + 1):
            if validate:
                x, y, df = next(gen)
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
                bb_list.append(df)
            else:
                x, df = next(gen)
                pred = self.model.predict(
                    x,
                    callbacks=None,
                    max_queue_size=1,
                    workers=1,
                    use_multiprocessing=False,
                    verbose=1
                )
                predict_list.append(pred)
                bb_list.append(df)

#        predict_li = [p for f in predict_list for p in f]
        bb_array = np.concatenate(bb_list, axis=0)
        #17_2c
        pred_list1, score_list1, taxa_list1 = get_tuple_list([predict_list[f][i][0]*bb_list[f][i][5] 
                                                 for f in range(len(predict_list)) for i in range(len(predict_list[f]))],0.99,0.54,0.09)
        #17_2
        pred_list2, score_list2, taxa_list2 = get_tuple_list([i[1] for f in predict_list for i in f],0.985,0.92,0.71)
        #17
        pred_list3, score_list3, taxa_list3 = get_tuple_list([i[2] for f in predict_list for i in f],0.98,0.89,0.71)
        #13
        pred_list4, score_list4, taxa_list4 = get_tuple_list([i[3] for f in predict_list for i in f],0.96,0.91,0.6)
        #13_2
        pred_list5, score_list5, taxa_list5 = get_tuple_list([i[4] for f in predict_list for i in f],0.99,0.96,0.67)
#        predict_bayes_li = [p for f in predict_bayes_list for p in f]
        all_array = np.concatenate([bb_array,np.array(pred_list1)[:,None],np.array(score_list1)[:,None],np.array(taxa_list1)[:,None],
                                    np.array(pred_list2)[:,None],np.array(score_list2)[:,None],np.array(taxa_list2)[:,None],
                                    np.array(pred_list3)[:,None],np.array(score_list3)[:,None],np.array(taxa_list3)[:,None],
                                    np.array(pred_list4)[:,None],np.array(score_list4)[:,None],np.array(taxa_list4)[:,None],
                                    np.array(pred_list5)[:,None],np.array(score_list5)[:,None],np.array(taxa_list5)[:,None]], axis=1)
#        assert(len(predict_li) > 0)
#        assert(len(predict_bayes_li) > 0)
        # check thresholds
#        pred_list, score_list, taxa_list = get_tuple_list(predict_li,0.97,0.52,0.08)
#        pred_bayes_list, score_bayes_list, taxa_bayes_list = get_tuple_list(predict_bayes_li,0.97,0.52,0.08)

#        return index, bb_list, pred_list, score_list, taxa_list, pred_bayes_list, score_bayes_list, taxa_bayes_list
        if validate:
            return all_array, label_list
        else:
            return all_array
