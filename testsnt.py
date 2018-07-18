import sonnet as snt
import tensorflow as tf

a = snt._resampler(tf.constant([0.]),tf.constant([0.]))
print(a)