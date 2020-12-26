import os
import subprocess
import time
import numpy as np
import csv
import h5py
import sys
from pathlib import Path

def extract_audio_features(pathIn, pathOut):
    config_path = current_directory
    conf_filepath = os.path.join(config_path, 'emobase2010_320_40.conf')
    count = 0

    # Use OpenSmile to extract audio features and save them in .csv files
    for filename in os.listdir(pathIn):
        print(count)
        print("Extracting audio from this clip: ", filename)

        if (filename.endswith(".wav")):  # or .avi, .mpeg, whatever.
            findMP4 = filename.find('.wav')
            csv_filename = filename[0:findMP4] +'.csv'

            audio_filepath = os.path.join(pathIn,filename)
            csv_filepath = os.path.join(pathOut, csv_filename)
            command = ["/Users/marius/opensmile/build/progsrc/smilextract/./SMILExtract", "-C", conf_filepath, "-I", audio_filepath, "-O", csv_filepath]
            subprocess.call(command)
        else:
            continue
        count += 1
    return 0

def average_extracted_features(pathOut):
    # List of csv output files
    csv_output_file_list = []
    for filename in os.listdir(pathOut):
        csv_output_file_list.append(filename)
    csv_output_file_list.sort()

    all_average_aud_feature =  []
    print('csv_output_file_list: ', csv_output_file_list)
    for path, folder, files in os.walk(audio_folder):
        files.sort()

        count = 0
        all_average_aud_feature = []
        for i in range(len(files)):
            filename = files[i]
            print("count: ", count, " filename: ", filename)
            clipTitle = filename.replace(".wav", "")
            for csv_openSmile_output_file in csv_output_file_list:
                csv_openSmile = csv_openSmile_output_file.replace(".csv", "")
                if csv_openSmile == clipTitle:
                    count = 0
                    clip_aud_feature = []
                    with open(os.path.join(pathOut, csv_openSmile_output_file), 'r') as csvFile:
                        csvReader = csv.reader(csvFile, delimiter=',')
                        next(csvReader) # skip the header
                        for i, row in enumerate(csvReader):  # csvReader:
                            feature = np.array(list(map(float, row[1:])))
                            clip_aud_feature.append(feature)
                            print(' ')
                    csvFile.close()
                    clip_aud_feature = np.vstack(clip_aud_feature)

                    average_clip_aud_feature = np.mean(clip_aud_feature, axis=0)

                    all_average_aud_feature.append(average_clip_aud_feature)
                    break
            count += 1
        # convert a list to an array
        all_average_aud_feature = np.array(all_average_aud_feature)
    return  all_average_aud_feature

if __name__ == "__main__":
    folder = sys.argv[1]
    del sys.argv[1]

    current_directory = Path(os.path.abspath(os.path.join(__file__, os.pardir)))
    vimotion = Path(current_directory).parent.parent

    path = current_directory

    start_time = time.time()
    print("Processing .......")

    audio_folder = os.path.join(vimotion, 'static/audioClips' + folder + '/audio')
    csv_output_folder = os.path.join(vimotion, 'static/audioClips' + folder)

    extract_audio_features(audio_folder, csv_output_folder)
    avg_features = average_extracted_features(csv_output_folder)

    # Write all average_aud_feature in a .h5 file
    h5filename = "emobase2010_320_40_features.h5"
    h5file = h5py.File(os.path.join(csv_output_folder, h5filename), mode='w')
    h5file.create_dataset('data', data=np.array(avg_features, dtype=np.float32))
    h5file.close()

    print('Extracting time: ', time.time()-start_time)
    start_time = time.time()

