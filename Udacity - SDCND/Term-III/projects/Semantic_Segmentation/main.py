import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    #Load VGG-16 model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    default_graph = tf.get_default_graph()

    input_image = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    #Apply scaling on outputs of 3 and 4 pooling layers as to further reduce loss
    vgg_layer3_out = tf.multiply(vgg_layer3_out, 0.0001, name='layer3_out_scaled')
    vgg_layer4_out = tf.multiply(vgg_layer4_out, 0.01, name='layer4_out_scaled')

    #Decoder portion begins
    # Layer 1: 1x1 convolution to reduce filters from 4096 to num_classes (2 in this case)
    conv_7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1),
                        padding='same',
                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    # Layer 2: Connection between con_7_1x1 and skip connection from layer 4
    #    i. First, con_7_1x1 needs to be upsampled by 2 to match with dimensions of layer 4
    #    ii. Then 1x1 convolution on layer 4 brings down the number of filters to num_classes (2 in this case)
    #    iii. skip layer is created with additive connection from layer 4 and conv_7_1x1
    upsample_2x_1 = tf.layers.conv2d_transpose(conv_7_1x1, num_classes, 4, strides=(2, 2),
                        padding='same',
                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    conv_4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1),
                        padding='same',
                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    skip_layer_4 = tf.add(upsample_2x_1, conv_4_1x1)

    # Layer 3: Connection between skip_layer_4 and skip connection from layer 3
    #    i. First, skip_layer_4 needs to be upsampled by 2 to match with dimensions of layer 3
    #    ii. Then 1x1 convolution on layer 3 brings down the number of filters to num_classes (2 in this case)
    #    iii. skip layer is created with additive connection from layer 3 and skip_layer_4
    upsample_2x_2 = tf.layers.conv2d_transpose(skip_layer_4, num_classes, 4, strides=(2, 2),
                        padding='same',
                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    conv_3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1),
                        padding='same',
                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    skip_layer_3 = tf.add(upsample_2x_2, conv_3_1x1)

    # Layer 4: Upsampler to regain input image. skip_layer_3 needs to be upsampled by 8 to match original image
    #            spatial dimensions
    upsample_8x = tf.layers.conv2d_transpose(skip_layer_3, num_classes, 16, strides=(8, 8),
                        padding='same',
                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    return upsample_8x
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    #Reshape logits to 2D tensor
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    #Reshape labels to 2D tensor
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    #Calculate cross entropy loss after applying soft max
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    #Collate regularizer losses encountered on addition of regularization on each decoder layer
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1  # Choose an appropriate one.

    #Calculate total loss
    total_loss = cross_entropy_loss + (reg_constant * sum(reg_losses))

    #Use Adam optimizer for training
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss)

    return logits, train_op, total_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("Training epoch: ", epoch + 1)
        for image, label in get_batches_fn(batch_size):
            training_op_result, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5,
                               learning_rate: 0.0008})
        print("Training Loss is: ", loss)
        print("------------------")
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out_scaled, layer4_out_scaled, layer7_out = load_vgg(sess, vgg_path)
        output = layers(layer3_out_scaled, layer4_out_scaled, layer7_out, num_classes)

        # Train NN using the train_nn function
        num_epochs = 25
        batch_size = 5

        #TF placeholders for label and learning_rate. Keep probability is already availble in the VGG graph
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                    correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    run()
