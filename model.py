"""
DCGAN to synthesize the log-magnitude and phase of people saying words.
This contains both the model (discriminator and generator), and also the training.
Useful network operations (such as convolutional or deconvolutional layers) are defined in ops.py.
Audio operations are defined in audio_utils.py.

Created by Felix Hafner, last edited on June 5th, 2018.
Praktikum for the embedded intelligence for healtcare and wellbeing chair.
University of Augsburg (UNIA).
"""

import numpy as np
import tensorflow as tf
import time
from random import randint
import os
from ops import batch_norm, conv_with_relu, deconv, fc
from audio_utils import spec2wav_file


SERVER = True       # To quickly change between training on a server or on your local machine.


# define Constants and Hyperparameters:
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("print_info", "-", '')       # any additional info, will be printed in beginning
tf.app.flags.DEFINE_string('save_name', "7", '')        # specifies the directory the output will be saved in
tf.app.flags.DEFINE_float('D_learning_rate', 0.001, '')
tf.app.flags.DEFINE_float('G_learning_rate', 0.002, '')
tf.app.flags.DEFINE_boolean("save_whole_batch", False, '')
tf.app.flags.DEFINE_integer("n_d_train", 1, '')
tf.app.flags.DEFINE_integer("n_g_train",1,'')
tf.app.flags.DEFINE_float("dropout_rate", 0.4, '')      # not used currently
tf.app.flags.DEFINE_boolean("gaussian", False, '')      # noise: Gaussian or uniform
tf.app.flags.DEFINE_integer("kernel_size", 5, '')
tf.app.flags.DEFINE_float("gen_threshold", -100,'')     # turned off
tf.app.flags.DEFINE_float("disc_threshold", -100, '')   # turned off
tf.app.flags.DEFINE_float("logmag_noise", 0.5,'')
tf.app.flags.DEFINE_float("phase_noise", 0.1, '')
tf.app.flags.DEFINE_integer("clip", 7, '')              # magnitude will be clipped [-clip, clip]
tf.app.flags.DEFINE_boolean("save_real", True, '')      # save inputs, useful for debugging purposes

if SERVER:
    # data_path: This is where the data is located. 
    # the generated data will be saved at data_path/save_name.
    # data_path/file_name is a list of all training-data paths.
    tf.app.flags.DEFINE_string("data_path", "/nas/student/FelixHafner/Data/", '')
    tf.app.flags.DEFINE_string("file_name", "/training_specs_list.txt", '')
    tf.app.flags.DEFINE_integer("batch_size", 128, '')
    tf.app.flags.DEFINE_integer('n_iterations', 80001, '')
    tf.app.flags.DEFINE_integer('save_iteration', 2000, '')     # specifies how often the output shall be saved.
    tf.app.flags.DEFINE_integer('loss_print_iteration', 100, '')# specifies how often the loss shall be printed.
    tf.app.flags.DEFINE_integer("training_size", 64000, '')

else:
    tf.app.flags.DEFINE_string("data_path",
                               "/Users/felixhafner/Google Drive/sonstiges/"
                               "Programmieren/Python/Praktikum Projects/Data", '')
    tf.app.flags.DEFINE_string("file_name", "/training_spec_list.txt", '')
    tf.app.flags.DEFINE_integer("batch_size", 8, '')
    tf.app.flags.DEFINE_integer('n_iterations', 201, '')
    tf.app.flags.DEFINE_integer('save_iteration', 1, '')
    tf.app.flags.DEFINE_integer('loss_print_iteration', 1, '')
    tf.app.flags.DEFINE_integer("training_size", 200, '')


def discriminator(spectrogram):
    """
    Defines the graph for the discriminator.
    Output should be one if audio is a real sample, zero if it is generated.
    :param spectrogram: The input: Either a real or generated audio.
    :return: the output is a single neuron with value [0,1]
    """

    with tf.variable_scope("d_", reuse=tf.AUTO_REUSE) as scope:

        # Add noise, and consider that input is [-1,1], so noise has to be scaled down
        logmag,phase = tf.unstack(spectrogram,axis=3)
        logmag = logmag + tf.random_normal(tf.shape(logmag),
                                                         stddev=(float(FLAGS.logmag_noise) / FLAGS.clip))
        phase = phase + tf.random_normal(tf.shape(phase), stddev = FLAGS.phase_noise / 2*np.pi)
        net = tf.stack([logmag, phase], axis=3)

        # Define the rest of the graph:
        net = conv_with_relu(net, [5, 5, 2, 64], 'conv1', False)
        net = conv_with_relu(net, [FLAGS.kernel_size, FLAGS.kernel_size, 64, 128], 'conv2', False)
        net = conv_with_relu(net, [FLAGS.kernel_size, FLAGS.kernel_size, 128, 256], 'conv3', False)
        net = conv_with_relu(net, [FLAGS.kernel_size,FLAGS.kernel_size, 256, 512], 'conv4', False)
        net = conv_with_relu(net, [FLAGS.kernel_size,FLAGS.kernel_size,512, 512], 'conv5', False)
        net = tf.reshape(net, [-1, net.shape[1] * net.shape[2] * net.shape[3]])
        net = fc(net, 1, 'fc1')
        return tf.nn.sigmoid(net), net


def generator(inputs):
    """
    Defines the graph for the generator.
    Takes input (random samples from uniform distribution) and returns a Spectrogram.
    :param inputs: A vector with random entries (either from a uniform or a gaussian distribution).
    :return: A generated audio sample, that hopefully fools the discriminator.
    """

    with tf.variable_scope("g_") as scope:
        net = fc(inputs, 28672, 'fc1')
        net = tf.reshape(net, [-1, 4, 7, 1024])
        net = tf.nn.relu(net)
        net = deconv(net, [FLAGS.kernel_size,FLAGS.kernel_size,1024, 1024], 'deconv1')
        net = batch_norm(True, net, 'bn2')
        net = tf.nn.relu(net)
        net = deconv(net, [FLAGS.kernel_size, FLAGS.kernel_size, 512, 1024], 'deconv2')
        net = batch_norm(True, net, 'bn3')
        net = tf.nn.relu(net)
        net = deconv(net, [FLAGS.kernel_size, FLAGS.kernel_size, 256, 512], 'deconv3')
        net = batch_norm(True, net, 'bn4')
        net = tf.nn.relu(net)
        net = deconv(net, [FLAGS.kernel_size, FLAGS.kernel_size, 128, 256], 'deconv4')
        net = batch_norm(True, net, 'bn5')
        net = tf.nn.relu(net)
        net = deconv(net, [FLAGS.kernel_size, FLAGS.kernel_size, 2, 128], 'deconv5')
        net = net[:, :98, :201, :]

        # Map logmag and phase to [-1,1]
        logmag, phase = tf.unstack(net, axis=3)
        phase = tf.nn.tanh(phase)
        logmag = tf.nn.tanh(logmag)
        net = tf.stack([logmag, phase], axis=3)

    return net


def load_data(file_list, path, number=-1):
    """
    loads and returns all data (spectrograms) from file_list.
    :param file_list: Contains a list of all the files to be loaded.
    :param path: Files can be found here.
    :param number: If number > -1, it returns that many spectrograms.
    :return: the data.
    """

    start_time = time.time()
    print ("-"*40)
    print "starting to load all data..."
    file_list = [f.strip() for f in file_list if f.strip()]
    length = number
    if number == -1:
        length = len(file_list)

    data = np.empty([length, 98, 201,2])
    for index, f in enumerate(file_list):

        if index % 5000 == 0 and index != 0:
            print "[loading data] ", index
        if index == length:
            break
        spectrogram = np.load(f)        # maybe np.load(data + f), depending on how the .txt file looks.

        # Check if there are any problems with the loaded data:
        if (np.isinf(np.sum(spectrogram)) or spectrogram.shape != (98, 201, 2)):
            print ("isinf or shape is wrong!")
            break

        # Clip logmag and map to [-1,1], Map phase from [0,2*pi] to [-1,1]:
        logmag, ph = np.dsplit(spectrogram, spectrogram.shape[-1])
        logmag = np.clip(logmag, -1*FLAGS.clip, FLAGS.clip) / FLAGS.clip
        ph = ph/np.pi - 1
        spectrogram = np.dstack((logmag,ph))
        data[index] = spectrogram

    print "done loading data. [%s seconds]" % (time.time() - start_time)
    return data


def get_batch(data):
    """
    Returns a random batch.
    :param data: Array containing all the data
    :return: the random batch.
    """

    indices = np.random.randint(low=0, high=FLAGS.training_size, size=[FLAGS.batch_size])
    return data[indices]


def print_summary():
    """
    Prints hyperparameters and parameters. Should be called at the start so that later confusion can be avoided.
    """

    print ("-"*40)
    print ("Info about architecture and parameters:")
    print FLAGS.print_info
    print "D learning_rate:", FLAGS.D_learning_rate
    print "G learning_rate:", FLAGS.G_learning_rate
    print "save_name:", FLAGS.save_name
    print "data_path:", FLAGS.data_path
    print "file_name:", FLAGS.file_name
    print "batch_size:", FLAGS.batch_size
    print "number of iterations:", FLAGS.n_iterations
    print "save iteration:", FLAGS.save_iteration
    print "Gaussian:", FLAGS.gaussian
    print "#Training of Discrim: ", FLAGS.n_d_train
    print "#Training of Generator: ", FLAGS.n_g_train
    print "kernel size:", FLAGS.kernel_size
    print "Generator Threshold:", FLAGS.gen_threshold
    print "Discriminator Threshold:", FLAGS.disc_threshold
    print "logmag noise:", FLAGS.logmag_noise
    print "phase noise:", FLAGS.phase_noise
    print "clipping input:", FLAGS.clip
    print "saving real data:", FLAGS.save_real
    print ("-"*40)


def gen_random_input(size):
    """
    Generates size many random values, either from uniform or gaussian distribution, as specified in FLAGS.gaussian
    :param size: specifies the number of random values
    :return: the random input
    """

    if (FLAGS.gaussian):
        mu, sigma = 0, 1
        return np.random.normal(mu,sigma, [size,100]).astype(np.float32)
    else:
        return np.random.uniform(-1, 1, [size, 100]).astype(np.float32)


def save_generated_audio(output, iteration):
    """
    Saves the generated audio (either just one sample or an entire batch, specified in FLAGS.save_whole_batch.
    May also save real input, as specified in FLAGS.save_real.
    When saving an entire batch, the function creates a new directory.
    :param output: the generated output that will be saved.
    :param iteration: is only needed for the name of the ouptut file.
    :return: nothing.
    """
    
    def map_back(spectrogram):
        """
        Map logmag from [-1,1] to [-clip, clip] and phase from [-1,1] to [0,2*pi]
        """
        logmag, ph = np.dsplit(spectrogram, spectrogram.shape[-1])
        logmag = logmag * FLAGS.clip
        ph = (ph + 1) * np.pi
        return np.dstack((logmag, ph))

    if (FLAGS.save_whole_batch):
        # create directory for batch
        extended_path = FLAGS.data_path+FLAGS.save_name+"/batch"+str(iteration).zfill(5)
        if "/batch"+str(iteration).zfill(5) not in os.listdir(FLAGS.data_path + FLAGS.save_name):
            os.mkdir(extended_path)

        # save entire output (generated):
        for index,generated_output in enumerate(output):
            np.save(extended_path+"/"+(str(index).zfill(3)), map_back(generated_output))
            spec2wav_file(extended_path+"/"+(str(index).zfill(3))+ ".npy")

        # save entire output (real):
        if FLAGS.save_real:
            random_input  = get_batch(data)
            for index, real_input in enumerate(random_input):
                np.save(FLAGS.data_path+FLAGS.save_name+"/real"+str(iteration).zfill(5), map_back(real_input))
                spec2wav_file(FLAGS.data_path+FLAGS.save_name+"/real"+str(iteration).zfill(5)+".npy")

    else:
        # save one output (generated):
        extended_path = FLAGS.data_path+FLAGS.save_name+"/"+str(iteration).zfill(5)
        generated_output = output[randint(0,FLAGS.batch_size-1)]

        np.save(extended_path, map_back(generated_output))
        spec2wav_file(extended_path + ".npy")

        # save one input (real):
        if FLAGS.save_real:
            real_input = get_batch(data)[0]

            np.save(FLAGS.data_path+FLAGS.save_name+"/real"+str(iteration).zfill(5), map_back(real_input))
            spec2wav_file(FLAGS.data_path+FLAGS.save_name+"/real"+str(iteration).zfill(5)+".npy")


if __name__ == "__main__":

    # load paths to data into lines
    with open(FLAGS.data_path + FLAGS.file_name) as f:
        lines = f.read().splitlines()

    # remove comments and load data
    lines = [line for line in lines if not line.startswith("#")]
    data = load_data(lines, FLAGS.data_path, FLAGS.training_size)

    print_summary()

    # create the folder for the output
    if FLAGS.save_name not in os.listdir(FLAGS.data_path):
        os.mkdir((FLAGS.data_path + FLAGS.save_name))

    # create Graph
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(shape=[None, 98, 201, 2], dtype=tf.float32)          # input discriminator
        z = tf.placeholder(shape=[None, 100], dtype=tf.float32)                 # input generator

        d_out_real, d_out_real_logits = discriminator(x)
        g_out = generator(z)
        d_out_fake, d_out_fake_logits = discriminator(g_out)

        # TODO: why
        """
        #d_out_real will be used, d_out_fake won't!. This will print out the outputs of the discriminator
        d_out_real = tf.Print(d_out_real, [d_out_real], message="d_out_real")
        d_out_real_logits = tf.Print(d_out_real_logits, [d_out_real_logits], message="d_out_real_logits")
        d_out_fake = tf.Print(d_out_fake, [d_out_fake], message="d_out_fake")
        d_out_fake_logits = tf.Print(d_out_fake_logits, [d_out_fake_logits], message="d_out_fake_logits")
        """

        # define loss for discriminator and generator:
        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_fake_logits,
                                                              labels=tf.zeros_like(d_out_fake))
        d_loss_fake = tf.reduce_mean(d_loss_fake)
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_real_logits,
                                                              labels=tf.ones_like(d_out_real))
        d_loss_real = tf.reduce_mean(d_loss_real)

        d_loss = d_loss_real + d_loss_fake

        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_fake_logits, labels=tf.ones_like(d_out_fake))
        g_loss = tf.reduce_mean(g_loss)

        train_variables = tf.trainable_variables()
        d_vars = [var for var in train_variables if 'd_' in var.name]
        g_vars = [var for var in train_variables if 'g_' in var.name]

        # training:
        d_optimizer = tf.train.AdamOptimizer(FLAGS.D_learning_rate).minimize(d_loss, var_list=d_vars)

        g_optimizer = tf.train.AdamOptimizer(FLAGS.G_learning_rate).minimize(g_loss, var_list=g_vars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        print "starting to train..."

        g_loss_avg = []
        d_loss_fake_avg = []
        d_loss_real_avg = []
        for iteration in xrange(FLAGS.n_iterations):

            # train discriminator:
            for _ in xrange(FLAGS.n_d_train):
                random_input = gen_random_input(FLAGS.batch_size)
                train_x = get_batch(data)
                loss_r, loss_f, _ = sess.run([d_loss_real, d_loss_fake, d_optimizer],
                                             feed_dict={x: train_x, z: random_input})
                d_loss_real_avg.append(loss_r)
                d_loss_fake_avg.append(loss_f)

                if (loss_r, loss_f < FLAGS.disc_threshold):
                    break

            # train the generator:
            for _ in xrange(FLAGS.n_g_train):
                random_input = gen_random_input(FLAGS.batch_size)
                loss, _ = sess.run([g_loss, g_optimizer], feed_dict={z: random_input})

                g_loss_avg.append(loss)
                if (loss < FLAGS.gen_threshold):
                    break

            # print loss
            if iteration % FLAGS.loss_print_iteration == 0:
                print ("-" * 40)
                print "iteration ", iteration, ". Batches trained: ", iteration*(FLAGS.n_g_train+FLAGS.n_d_train)
                print "discriminator (Real) loss (avg): ", np.mean(d_loss_real_avg)
                print "discriminator (Fake) loss (avg): ", np.mean(d_loss_fake_avg)
                print "generator loss (avg): ", np.mean(g_loss_avg)
                print "training for [%s seconds]" % (time.time() - start_time)
                g_loss_avg = []
                d_loss_fake_avg = []
                d_loss_real_avg = []

            # save sample output:
            if iteration % FLAGS.save_iteration == 0:
                print "saving sample generator output..."

                # evaluate output of the generator.
                random_in = gen_random_input(FLAGS.batch_size)
                generator_output = np.array(sess.run(g_out, feed_dict={z: random_in}))

                save_generated_audio(generator_output, iteration)

        # else is taken whenever a for loop terminates 'normally', i.e. without breaks.
        else:
            print "done training. [%s seconds]" % (time.time() - start_time)
            print "time for one iteration: %s seconds" % ((time.time()-start_time)/FLAGS.n_iterations)


