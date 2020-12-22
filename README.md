# Vimotion
Vimotion is a Flask-based Webapp for emotion based matching of audio and video.

## Setup Vimotion
**Requirements** (packages are available to install with pip):
- Python 3.7 or before
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


Adjust the path, or if needed the full command, in "feature_extraction_emobase2010_320_40.py" line 26.


If you don't use Python3.7 adjust the commands in vimotion.py line 408-424.


If you get this error on linking a YouTube-video:
"urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1045)>"
from the Vimotion directory run this command:
"open /Applications/Python\ 3.7/Install\ Certificates.command"


Vimotion is now ready to run!

## Change data sets

#### How to add file(s) to the audio data set:
- create a folder (not inside Vimotion!)
- copy or move the audio files you want to add into this folder
- audio files should be at least 40 seconds!
- make sure only the audio files which should be added are in this folder
- set path of this folder in "expand_audio_dataset.py"
- run "expand_audio_dataset.py"
- when running is completed, add information of title (second column) and artist (third column) to the corresponding lines in metadata.csv (static/datasets/audio)


#### How to add file(s) to the video data set:
- create a folder (not inside Vimotion!)
- copy or move the video files you want to add into this folder
- video files should be at least 40 seconds and have to have at least 12.8 frames per second!
- make sure only the video files which should be added are in this folder
- set path of this folder in "expand_video_dataset.py"
- run "expand_video_dataset.py"
- when running is completed, add information of title (second column) and source (third column, i.e. YouTube link) to the corresponding lines in metadata.csv (static/datasets/audio)


#### How to remove file(s) from audio data set:
- list the file names to delete in the "files_to_remove" array in "reduce_audio_dataset.py" WITHOUT the file ending (i.e. "2" instead of "2.mp3")
- run "reduce_audio_dataset.py"


#### How to remove file(s) from video data set:
- list the file names to delete in the "files_to_remove" array in "reduce_video_dataset.py" WITHOUT the file ending (i.e. "Mulan Avalanche Scene" instead of "Mulan Avalanche Scene.mp4")
- run "reduce_video_dataset.py"
