# Vimotion

Reuqirements (packages are available to install with pip):
- Python 3.5
- OpenSMILE
- Flask
- Pytube
- OpenCV
- FFmpeg
- FFmpeg-python
- Mutagen
- Pydub
- Tensorflow 1.14.0
- Scipy
- Torch torchvision
- Resampy
- Soundfile
- Seaborn
- Get-video-properties
- Tqdm
- Imageio
- --no-binary=h5py h5py
- Pandas


Adjust the paths inside the following files:
- audio_model_arousal.py
- audio_model_valence.py
- video_model_arousal.py
- video_model_valence.py
- feature_extraction_ResNet50.py
- feature_extraction_I3D_RGB.py
- feature_extraction_FlowNetS.py
- feature_extraction_VGGish.py
- feature_extraction_emobase2010_320_40.py


Add file(s) to the audio data set:
- create a folder (not inside Vimotion!)
- copy or move the audio files you want to add into this folder
- audio files should be at least 40 seconds!
- make sure only the audio files which should be added are in this folder
- set path of this folder in "expand_audio_dataset.py"
- run "expand_audio_dataset.py"
- when running is completed, add information of title (second column) and artist (third column) to the corresponding lines in metadata.csv (static/datasets/audio)


Add file(s) to the video data set:
- create a folder (not inside Vimotion!)
- copy or move the video files you want to add into this folder
- video files should be at least 40 seconds and have to have at least 12.8 frames per second!
- make sure only the video files which should be added are in this folder
- set path of this folder in "expand_video_dataset.py"
- run "expand_video_dataset.py"
- when running is completed, add information of title (second column) and source (third column, i.e. YouTube link) to the corresponding lines in metadata.csv (static/datasets/audio)


Remove file(s) from audio data set:
- list the file names to delete in the "files_to_remove" array in "reduce_audio_dataset.py" WITHOUT the file ending (i.e. "2" instead of "2.mp3")
- run "reduce_audio_dataset.py"


Remove file(s) from video data set:
- list the file names to delete in the "files_to_remove" array in "reduce_video_dataset.py" WITHOUT the file ending (i.e. "Mulan Avalanche Scene" instead of "Mulan Avalanche Scene.mp4")
- run "reduce_video_dataset.py"
