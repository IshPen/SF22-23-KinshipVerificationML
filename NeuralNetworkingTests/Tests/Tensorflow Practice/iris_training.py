import tensorflow as tf
import numpy as np

training_set = tf.data.Dataset(
        filename='IRIS_TRAINING',
        target_dtype=np.int,
        features_dtype=np.float32)
print(training_set)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename='IRIS_TEST',
        target_dtype=np.int,
        features_dtype=np.float32)