import tensorflow as tf
import numpy as np
import abc
import dill



class BaseModel(abc.ABC):
    '''
    This class works as a wrapper to have a single interface to several
    models and machine learning packages. This will hide the complexity
    of the different ways in which the algorithms are used. This is inteded
    to be used with the xarray package.
    '''
    def __init__(self):
        '''
        This is init of Basemodel
        '''
        pass

    def fit(self, X, y):
        '''
        This method will train the classifier with given data.
        '''
        NotImplementedError('Children of BaseModel need to implement their own fit method')


    def predict(self, X):
        '''
        When the model is created, this method lets the user predict on unseen data.
        '''
        NotImplementedError('Children of BaseModel need to implement their own predict method')


    def save(self, filepath):
        '''
        Write entire object to file
        '''
        with open(filepath, 'wb') as dst:
            dill.dump(self, dst)


    @staticmethod
    def load(filepath):
        '''
        Read object from file
        '''
        with open(filepath, 'rb') as src:
            obj = dill.load(src)
        return obj


def inception_features(target_size):
    inception_features = tf.keras.applications.InceptionResNetV2(input_shape=(target_size,target_size,3), include_top=False, weights='imagenet')
    inception_features.trainable = False
    model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(target_size,target_size,3), name='images_input'),
            inception_features])

    return model


def head_for_inception_model(lambd1,nclasses,activation='sigmoid'):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(8,8,1536), name='features_input'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(nclasses, activation=activation,kernel_regularizer=tf.keras.regularizers.l1(lambd1))
    ])
    
    return model

def dense_coo_layer(inp,lambd1,activation='sigmoid'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(51, activation=activation, kernel_regularizer=tf.keras.regularizers.l1(lambd1))
    ])
    res = model(inp)
    return res

def join_models(model1,model2):
    model = tf.keras.Sequential([
            model1,
            model2
    ])

    return model

def get_tuple_list(predict_li,threshold1=None,threshold2=None,threshold3=None):
    if threshold1 is not None:
        tuple_list = [(np.argmax(predict_li[ind]), predict_li[ind][np.argmax(predict_li[ind])],'species')
                      if predict_li[ind][np.argmax(predict_li[ind])] > threshold1
                      else (np.argmax(predict_li[ind]), predict_li[ind][np.argmax(predict_li[ind])],'genus')
                      if predict_li[ind][np.argmax(predict_li[ind])] > threshold2
                      else (np.argmax(predict_li[ind]), predict_li[ind][np.argmax(predict_li[ind])],'family')
                      if predict_li[ind][np.argmax(predict_li[ind])] > threshold3
                      else (None, None, None)
                      for ind in range(len(predict_li)) if predict_li[ind][np.argmax(predict_li[ind])] >= 0.0
                     ]
    else:
        tuple_list = [(np.argmax(predict_li[ind]), predict_li[ind][np.argmax(predict_li[ind])],'species')
                      for ind in range(len(predict_li)) if predict_li[ind][np.argmax(predict_li[ind])] >= 0.0
                     ]

    if len(tuple_list) > 0:
        pred_list = [int(tu[0]) if tu[0] is not None else tu[0] for tu in tuple_list]
        score_list = [tu[1] for tu in tuple_list]
        taxa_list = [tu[2] for tu in tuple_list]

    else:
        pred_list = []
        score_list = []
        taxa_list = []

    return pred_list, score_list, taxa_list

