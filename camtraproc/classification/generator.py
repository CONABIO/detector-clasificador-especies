import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import PIL.Image
import cv2
import math

class Features_generator_from_dataframe(tf.keras.utils.Sequence):

        def __init__(self, dataframe, directory, batch_size, x_col, target_size, seed, y_col=None):
            if seed:
                self.seed = seed
                self.df = shuffle(dataframe, random_state=seed)
            else:
                self.seed = None
                self.df = shuffle(dataframe)
            self.x_col = x_col
            self.y_col = y_col
            self.batch_size = batch_size
            self.dirpath = directory
            self.seed = seed
            self.target_size = target_size


        def __len__(self):
            return math.ceil(len(self.df) / self.batch_size)

        def __getitem__(self, idx):
            batch_x = np.array(self.df[self.x_col].iloc[idx * self.batch_size:(idx + 1) *
            self.batch_size])

            x = [self._crop_generator(x) for x in batch_x]
            indexes = self._get_index(x)
            if self.y_col is not None:
                batch_y = list(self.df[self.y_col[0]].iloc[idx * self.batch_size:(idx + 1) *
                    self.batch_size])
                return np.array(x)[indexes], self._get_list(batch_y,indexes), batch_x[indexes]
            else:
                return np.array(x)[indexes], batch_x[indexes]

        def on_epoch_end(self):
            if self.seed:
                self.df = shuffle(self.df, random_state=self.seed)
            else:
                self.df = shuffle(self.df)

        def _crop_generator(self,x_array):
            try:
                img = PIL.Image.open(self.dirpath + '/' + x_array[1]).convert("RGB"); img = np.array(img)
                return np.array(cv2.resize(img,(self.target_size,self.target_size))/255.)
            except Exception as e:
                return False

        def _get_index(self,batch_x):
            return [f for f in range(len(batch_x)) if batch_x[f] is not False]

        def _get_list(self,list1,indexes):
            return [list11[f] for f in range(len(list1)) if f in list(indexes[0])]

class Features_generator_for_feat_from_dataframe(tf.keras.utils.Sequence):

        def __init__(self, dataframe, directory, batch_size, x_col, target_size, seed):
            if seed:
                self.seed = seed
                self.df = shuffle(dataframe, random_state=seed)
            else:
                self.seed = None
                self.df = shuffle(dataframe)
            self.x_col = x_col
            self.batch_size = batch_size
            self.dirpath = directory
            self.seed = seed
            self.target_size = target_size


        def __len__(self):
            return math.ceil(len(self.df) / self.batch_size)

        def __getitem__(self, idx):
            batch_x = np.array(self.df[self.x_col].iloc[idx * self.batch_size:(idx + 1) *
            self.batch_size])

            x = [self._crop_generator(x) for x in batch_x]
            indexes = self._get_index(x)
            return np.array(x)[indexes], batch_x[indexes]

        def on_epoch_end(self):
            if self.seed:
                self.df = shuffle(self.df, random_state=self.seed)
            else:
                self.df = shuffle(self.df)

        def _crop_generator(self,x_array):
            try:
                img = PIL.Image.open(self.dirpath + '/' + x_array[1]).convert("RGB"); img = np.array(img)
                return np.array(cv2.resize(img,(self.target_size,self.target_size))/255.)
            except Exception as e:
                return False

        def _get_index(self,batch_x):
            return [f for f in range(len(batch_x)) if batch_x[f] is not False]

class Features_generator_ex_coo_dist_pot_from_dataframe(tf.keras.utils.Sequence):

        def __init__(self, dataframe, directory, batch_size, x_col, seed, y_col=None):
            if seed:
                self.seed = seed
                self.df = shuffle(dataframe, random_state=seed)
            else:
                self.seed = None
                self.df = shuffle(dataframe)
            self.x_col = x_col
            self.y_col = y_col
            self.batch_size = batch_size
            self.dirpath = directory
            self.seed = seed


        def __len__(self):
            return math.ceil(len(self.df) / self.batch_size)

        def __getitem__(self, idx):
            batch_x1 = np.array(self.df[self.x_col].iloc[idx * self.batch_size:(idx + 1) *
            self.batch_size])
            batch_x2 = list(self.df[self.x_col[6]].iloc[idx * self.batch_size:(idx + 1) *
            self.batch_size])
            batch_x3 = list(self.df[self.x_col[7]].iloc[idx * self.batch_size:(idx + 1) *
            self.batch_size])

            x = [self._load_generator(x) for x in batch_x1]
            coor1 = np.array([np.array(coor) for coor in batch_x2])
            coor2 = np.array([np.array(coor) for coor in batch_x3])
            indexes = self._get_index(x)
            if self.y_col is not None:
                batch_y = list(self.df[self.y_col[0]].iloc[idx * self.batch_size:(idx + 1) *
                self.batch_size])
                # x, y, xdf
                return [np.array(x)[indexes], coor1[indexes],
                        coor2[indexes]], self._get_list(batch_y,indexes), batch_x1[indexes]
            else:
                # x, xdf
                return [np.array(x)[indexes], coor1[indexes], coor2[indexes]], batch_x1[indexes]

        def on_epoch_end(self):
            if self.seed:
                self.df = shuffle(self.df, random_state=self.seed)
            else:
                self.df = shuffle(self.df)

        def _load_generator(self,x_array):
            try:
                feat = np.load(self.dirpath + '/' + x_array[1].split('.')[0] + '.npy')
                return feat
            except Exception as e:
                return False

        def _get_index(self,batch_x):
            return [f for f in range(len(batch_x)) if batch_x[f] is not False]

        def _get_list(self,list1,indexes):
            return [list11[f] for f in range(len(list1)) if f in list(indexes[0])]
