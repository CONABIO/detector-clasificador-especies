import tensorflow as tf

inception_features = tf.keras.applications.InceptionResNetV2(input_shape=(299,299,3), include_top=False, weights='imagenet')

inception_features.summary()
