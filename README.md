Synthezising Audio using Generative Adversarial Networks.


--------
Transforming the data:

1: Download the google speech command data set.
2: Pad every audio file to one second.
	This can be done with the function pad_all(src_dir) defined in audio_utils.py.
	This function will automatically iterate over every directory and every .wav and pad it, if necessary.
3: Transform every audio file into a spectrogram of shape (98,201,2). 
	This can be done with the function wav2spec(src_dir, dst_dir). 
	It will automatically create a .txt file containing the paths to every spectrogram.
	The name of the file will be dst_dir+"_list.txt".
4: Loading the data:
	This can be done with the function load_data(file_list, path, number) in model.py.
--------
Training the model:

Adjust the Flags. Most important ones are data_path and file_name (containting the paths).
Depending on how you generated the file, the function load_data either expects 
	spectrogram = np.load(f) or spectrogram = np.load(data+f)
Loading the data takes roughly 15 mintues, and the data is about 10gb big. If 15 mintues is not acceptable, 
it might be advantageous to once load all spectrograms into one numpy array and save this once.

--------

Future work:
The next steps to improve this model would be to figure out a more fitting network architecture 
(the main problem is generating the phase), or trying out if some other loss function (see wasserstein GANs, weight clipping and gradient penalty)
works better.
	
