"""
Some useful functions when dealing with audio.

Created by Gil Keren. Edited by Felix Hafner.
"""


import tensorflow as tf, scipy.io.wavfile, scipy.signal, numpy as np, wave, matplotlib.pyplot as plt
import os

CHUNK = 1024
CHANNELS = 1
RATE = 16000



def get_spectrogram(wavpath):
    """
    Takes a path to a wav file and computes the log-magnitude, the phase, and the number of samples
    from the audio
    :param wavpath: Path to the wav
    :return: Log-magnitude, phase, and number of samples
    """

    g = tf.Graph()
    with g.as_default():
        samplerate, samples = scipy.io.wavfile.read(wavpath)
        samples = samples / (max(abs(samples)) + 0.000001)
        samples = samples.astype(np.float32)
        assert samplerate == 16000
        assert len(samples.shape) == 1
        nsamples = len(samples)
        samples = tf.constant(np.reshape(samples, (1, -1)))
        stft = tf.contrib.signal.stft(samples, 400, 160, 400)
        mag = tf.abs(stft)
        log_mag = tf.log(mag)
        ph = tf.angle(stft)
        sess = tf.Session()
        logmagval, phval = sess.run([log_mag, ph])
        return logmagval, phval, nsamples


def waveform_from_spectrogram(log_mag, ph, path=''):
    """
    Takes the logmagnitude and phase and saves it as a wav file at path
    :param log_mag: The log magnitude
    :param ph: The phase
    :param path: The path where the audio will be saved
    :return: Nothing to return
    """

    mag = np.exp(log_mag)
    complex_stft = mag * (np.exp(1j * ph))
    g = tf.Graph()
    with g.as_default():
        stft_ph = tf.placeholder(dtype=tf.complex64, shape=[1, None, 201])  # [1, time, freq]
        samples = tf.contrib.signal.inverse_stft(stft_ph, 400, 160, 400)[0, :]

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0}))
        samples_np = sess.run(samples, {stft_ph: complex_stft})
        samples_np = samples_np / (max(abs(samples_np)) + 0.000001)
        if path:
            scipy.io.wavfile.write(path, 16000, samples_np)


def plot(values, colormap="jet"):
    """
    Helper function to plot the phase or the log-magnitude
    :param values:
    :param colormap:
    :return: Nothing to return
    """

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(values), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")

    plt.show()


def plot_l_mag(wavpath):
    """
    Takes the path to a wav file and plots the log-magnitude
    :param wavpath: The path to the audio
    :return: Nothing to return, just showing the plot
    """

    l_mag, phase, nsamples = get_spectrogram(wavpath)
    l_mag, phase = l_mag[0], phase[0]
    plot(l_mag)


def plot_phase_difference(wavpath):
    """
    Takes the path to a wav file and plots the phase difference
    :param wavpath: the path to the audio
    :return: Nothing to return, just showing the plot
    """

    logmag, phase, nsamples = get_spectrogram(wavpath)
    logmag, phase = logmag[0], phase[0]

    from matplotlib.colors import LinearSegmentedColormap
    colors = ["black", "white", "black"]
    cmap = LinearSegmentedColormap.from_list("", colors)
    #phase_difference = get_phase_difference(phase)
    phase_difference=phase
    plot(phase_difference, cmap)



def wav2spec(src_dir, dst_dir):
    """
    Converts all wav files to spectrograms. 
    Also writes the paths to all spectrograms into a .txt
    :param src_dir: Path to all the wav files.
    :param dst_dir: Converted spectrograms will be saved here.
    :return: -
    """
    
    g = tf.Graph()
    with g.as_default():
        samples_pl = tf.placeholder(shape=[1,16000], dtype=tf.float32)
        stft = tf.contrib.signal.stft(samples_pl, 400, 160, 400)
        mag_graph = tf.abs(stft)
        l_mag_graph = tf.log(mag_graph + 0.000001)
        phase_graph = tf.angle(stft)

        disregarded_folders = ['DS_Store', '@eaDir', '.DS_Store'];

        sess = tf.Session()
        with open(dst_dir + "_list.txt", "a") as f:

            #iterate over all audio (wav) files:
            for folder in os.listdir(src_dir):
                if not os.path.isdir(src_dir + '/' + folder):
                    continue
                print folder
                i = 0
                if folder in disregarded_folders:
                    continue
                for wav in os.listdir(src_dir + '/' + folder):
                    if (wav in disregarded_folders):
                        continue
                    if not (wav.endswith(".wav")):
                        continue
                    path = src_dir + '/' + folder + '/' + wav
                    if not os.path.isfile(path):
                        continue

                    #convert wav to spectrogram:
                    samplerate, samples = scipy.io.wavfile.read(path)
                    samples = samples / (max(abs(samples)) + 0.000001)
                    samples = samples.astype(np.float32)
                    assert samplerate == 16000
                    assert len(samples.shape) == 1
                    samples = np.reshape(samples, (1, -1))
                    if samples.shape != (1,16000):
                        continue
                    l_mag, phase = sess.run([l_mag_graph, phase_graph], feed_dict={samples_pl: samples})
                    phase = get_phase_difference(phase[0])
                    spectrogram = stack(l_mag[0], phase)

                    #save the spectrogram:
                    file_ending = "/" + folder + str(i).zfill(5)
                    np.save(dst_dir + file_ending, spectrogram)
                    f.write(dst_dir + file_ending + ".npy\n")

                    if i % 500 == 0:
                        print "iteration in folder: ", i
                    i += 1


def spec2wav_dir(src_dir):
    """
    Loads all spectrograms given in the directory src_dir, converts and saves them to wav files
    and saves them.
    Uses function spec2wav_file(spectrogram_path)
    :param src_dir: The path to the source directory, containing the spectrograms
    :return: Nothing, the wavs will be saved directly
    """

    for file_name in os.listdir(src_dir):
        if file_name.endswith(".npy"):
            spec2wav_file(src_dir + file_name)


def spec2wav_file(spectrogram_path):
    """
    Loads the spectrogram specified in spectrogram_path, converts it to a wav, and saves it
    at the same location with the same name, but .wav instead of .npy.
    Only works if path ends with .npy

    Assumes that the input spectrogram is using the phase difference!
    :param spectrogram_path: The path to the spectrogram. Has to end with ".npy"
    :return: Nothing, the wav will be saved directly
    """

    spectrogram = np.load(spectrogram_path)
    if (spectrogram.shape[2] == 2):     #Spectrogram consists out of phase and logmag
        l_mag, phase = np.dsplit(spectrogram, spectrogram.shape[-1])
        phase = get_phase_from_difference(phase)
    else:                               #Spectrogram only consists of logmag. Setting phase to zero
        l_mag = spectrogram
        phase = np.zeros(l_mag.shape)
    waveform_from_spectrogram(np.array([l_mag[:, :, 0]]), np.array([phase[:, :, 0]]), spectrogram_path[:-4] + ".wav")


def stack(l_mag, phase):
    """
    Takes the logmagnitude and the phase and stacks them together to a spectrogram
    :param l_mag: the log-magnitude, shape: [:,:,1]
    :param ph: the phase, shape: [:,:,1]
    :return: the spectrogram
    """

    a = np.dstack((l_mag,phase))
    return a


def unstack(spectrogram):
    """
    Takes a spectrogram and returns it's components
    :param stacked: The spectrogram
    :return: the logmag and the phase
    """

    return np.dsplit(spectrogram, spectrogram.shape[-1])


def get_phase_difference(phase):
    """
    Computes the difference between each time step of the phase.
    First time step stays the same!
    :param ph: the phase
    :return: The phase difference
    """

    return np.mod(np.append([phase[0]],phase[1:]-phase[:-1], axis=0), 2*np.pi)


def get_phase_from_difference(phase_difference):
    """
    Takes the phase difference and computes the original phase
    Still sometimes not correct, but close enough.
    :param ph: The phase difference
    :return: The original phase
    """

    phase = np.zeros_like(phase_difference)
    for i in xrange(201):
        x = phase_difference[0][i]
        phase[0][i] = x if x < np.pi else np.mod(x, np.pi) - np.pi
    for j in xrange(200):
        # look at i+1
        for i in xrange(97):
            x = phase_difference[i + 1][j] + phase[i][j]
            phase[i + 1][j] = x if x < np.pi else (np.mod(x, np.pi) - np.pi if x < 2 * np.pi else np.mod(x, 2 * np.pi))

    return phase


def pad_all(src_dir):
    """
    pads all audio (wav) files in the directories in src_dir to one second (with silence)
    :param src_dir: the path to the source directory
    :return: Nothing to return
    """

    for folder in os.listdir(src_dir):
        if not os.path.isdir(src_dir + '/' + folder):
            continue
        print folder
        i = 0
        if folder in ['DS_Store', '@eaDir', '.DS_Store']:       #directories/files that are created
            continue
        for wav in os.listdir(src_dir + '/' + folder):
            if (wav in ['.DS_Store', '@eaDir', 'DS_Store']):    #directories/files that are created
                continue
            if not (wav.endswith(".wav")):                      #check if file is actually a wav file
                continue
            path = src_dir + '/' + folder + '/' + wav
            if not os.path.isfile(path):
                continue

            samplerate, samples = scipy.io.wavfile.read(path)
            samples = samples.astype(np.float32)
            assert samplerate == 16000
            assert len(samples.shape) == 1
            samples = np.reshape(samples, (1, -1))

            if samples.shape != (1, 16000):
                #pad and copy:
                os.system("sox "+path+" "+path[:-4]+"_PA.wav pad 0 1 trim 0 =1.0")
                #delete old one
                os.system("rm "+path)
                print "padding"




if __name__ == '__main__':
    sess = tf.Session()

    data_path_server = "/nas/student/FelixHafner/Data/"
    data_path_home = "/Users/felixhafner/Google Drive/sonstiges/Programmieren/Python/Praktikum Projects/Data/"

    #pad_all(data_path_server+"/training_wav")
    plot_l_mag(data_path_home+"gen.wav")
    plot_phase_difference(data_path_home+"gen.wav")
    #wav2spec(data_path_server + "/training_wav", data_path_server + "/training_specs")
    #spec2wav_dir(data_path+"/DCGAN_output/", data_path+"/DCGAN_wavs/")
