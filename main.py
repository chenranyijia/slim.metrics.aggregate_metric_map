import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets import vgg

# Load the data



mnist = tf.contrib.learn.datasets.load_dataset("mnist")
images = mnist.train.images  # Returns np.array
images = images[:10,:]
images = np.reshape(images,[images.shape[0],28,28,-1])
images = np.concatenate([images] * 3, 3)
images = tf.image.resize_images(images, tf.constant([224,224]), method=tf.image.ResizeMethod.BILINEAR)
print images.shape

labels = np.asarray(mnist.train.labels, dtype=np.int64)

labels = labels[:10]
print labels.shape
# Define the network
predictions, _  = vgg.vgg_16(images, num_classes=10, spatial_squeeze=True)

predictions = tf.argmax(predictions, 1)

# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/mean_absolute_error": slim.metrics.streaming_mean_absolute_error(predictions, labels),
    "eval/mean_squared_error": slim.metrics.streaming_mean_squared_error(predictions, labels),
})

# Evaluate the model using 1000 batches of data:
num_batches = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for batch_id in range(num_batches):
        sess.run(names_to_updates.values())

    metric_values = sess.run(names_to_values.values())
    for metric, value in zip(names_to_values.keys(), metric_values):
        print('Metric %s has value: %f' % (metric, value))
