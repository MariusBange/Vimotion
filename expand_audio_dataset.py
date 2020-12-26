'''add audio files to audio data set'''
import os
import ffmpeg
from mutagen.mp3 import MP3
import subprocess
import shutil
import math
import json
import csv

audio_folder_path = "/Users/marius/Desktop/add" #Path of folder of the audio file(s) which should be added. IMPORTANT: Only audio files in this folder which should be added to the dataset!

if os.path.isdir(audio_folder_path):

    dataset = {}
    errors = []
    dataset_values = "static/datasets/audio/dataset_values.txt"

    with open(dataset_values) as f:
        dataset = json.load(f)
    f.close()

    files = [f for f in os.listdir(audio_folder_path) if os.path.isfile(os.path.join(audio_folder_path, f))]

    mp3_files = []
    folders = []
    files_to_delete = []
    copied_files = []

    ########## Convert files to .mp3 ##########
    for file in files:
        if file[-4:] != '.mp3':
            if file[-4:] not in ['.aac', '.flac', '.ogg', '.wav', '.wma']:
                continue
            input_audio = ffmpeg.input(os.path.join(audio_folder_path, file))
            dot_index = file.rfind('.')
            raw_filename = file[0:dot_index]
            filename = raw_filename
            i = 1

            while os.path.isfile(os.path.join(audio_folder_path, filename) + '.mp3'):
                i += 1
                filename = raw_filename + '(' + str(i) + ')'

            ffmpeg.output(input_audio, os.path.join(audio_folder_path, filename) + '.mp3').run()
            mp3_files.append(filename + '.mp3')
        else:
            mp3_files.append(file)

    ########## Copy files into dataset ##########
    for file in mp3_files:
        audio_string = file
        dot_index = audio_string.rfind('.')
        audio_string = audio_string[0:dot_index]

        filename = audio_string + '.mp3'
        i = 0
        while os.path.isfile('static/datasets/audio/' + filename):
            i += 1
            filename = audio_string + '(' + str(i) + ').mp3'
        shutil.copy(os.path.join(audio_folder_path, file), 'static/datasets/audio/' + filename)
        copied_files.append(filename)

    ########## Feature extraction for each file ##########
    for file in copied_files:
        mp3_file = MP3(os.path.join(audio_folder_path, file))
        duration = mp3_file.info.length

        if duration < 40:
            continue
        else:
            audio_string = file
            dot_index = audio_string.rfind('.')
            audio_string = audio_string[0:dot_index]
            folder = audio_string + '-clips(1)'
            i = 1

            while os.path.exists('static/audioClips/'+folder):
                i += 1
                folder = audio_string + '-clips(' + str(i) + ')'
            folders.append(folder)
            os.makedirs('static/audioClips/'+folder+'/audio')

            f = open("static/audioClips/"+folder+"/values.txt", "w")
            f.write(str(duration))
            f.close()

            ########## Split files into 5 second segments ##########
            wav_file = 'static/uploads/audio/' + audio_string + '.wav'
            subprocess.call(['ffmpeg', '-i', os.path.join(audio_folder_path, file), wav_file])
            segments = math.ceil(duration / 5)
            files_to_delete.append(wav_file)

            for i in range(0, segments):
                start_time = str(i * 5) + '.0'
                if i < segments - 1:
                    end_time = '5.0'
                else:
                    end_time = str(duration % 5)
                if float(end_time) > 1:
                    subprocess.call(['ffmpeg', '-ss', start_time, '-i', wav_file, '-c', 'copy', '-t', end_time, 'static/audioClips/' + folder + '/audio/' + audio_string + '(' + str(i) + ')' + '.wav'])

            ########## Extract features ##########
            os.system("python3.7 Model/VGGish/feature_extraction_VGGish.py '/" + folder + "'")
            os.system("python3.7 Model/OpenSMILE/feature_extraction_emobase2010_320_40.py '/" + folder + "'")
            os.system("python3.7 Model/AttendAffectNet/audio_model_arousal.py '/" + folder + "'")
            os.system("python3.7 Model/AttendAffectNet/audio_model_valence.py '/" + folder + "'")

            values = 'static/audioClips/' + folder + "/values.txt"

            duration = 0.0
            arousal = []
            valence = []

            with open(values) as f:
                duration = float(f.readline().strip())
                arousal_string = f.readline().strip()
                valence_string = f.readline().strip()

            arousal = [float(value.strip()) for value in arousal_string.split(', ')]
            valence = [float(value.strip()) for value in valence_string.split(', ')]
            for i in range(0, len(arousal)):
                if not isinstance(arousal[i], float) or not isinstance(valence[i], float):
                    errors.append(audio_string)
                    break
            if audio_string not in errors:
                dataset[audio_string] = {'duration': duration, 'arousal': arousal,'valence': valence}
            else:
                print("ERROR: The file ", audio_string, " could not be added due to an error while extracting features.")

    f = open(dataset_values, "w")
    dataset_updated = json.dumps(dataset)
    f.write(dataset_updated)
    f.close()

    ########## Add files to metadata.csv ##########
    metadata_rows = []

    with open('static/datasets/audio/metadata.csv', newline='') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            metadata_rows.append(row)

    for file in copied_files:
        metadata_rows.append([file, "", ""])

    with open('static/datasets/audio/metadata.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(metadata_rows)

    files_to_delete.append('static/audioClips/' + folder)

    modes = ["standard", "weighted", "mse"]

    for mode in modes:
        exponent = 2 if mode == "mse" else 1
        writing_weight_1 = 1
        writing_weight_2 = 1
        text_file_ending = ".txt" if mode == "standard" else "_weighted.txt"

        if mode == "mse":
            text_file_ending = "_mse.txt"

        ########## Update ordering by distance ##########
        top_level_keys = []
        ordering = {}
        keys_and_durations = []

        for key in dataset:
            keys_and_durations.append([5 * math.ceil(dataset[key]['duration']/5), key])
            ordering[key] = []

        keys_and_durations = sorted(keys_and_durations, key=lambda x: x[0], reverse=True)
        top_level_keys.append(keys_and_durations[0][1])
        start = 1

        for i in range(start, len(keys_and_durations)):
            if keys_and_durations[i][0] < keys_and_durations[i-1][0]:
                start = i
                break

            top_level_keys.append(keys_and_durations[i][1])

        ########## Iterate trough all dataset items ##########
        for i in range(start, len(keys_and_durations)):
            print(i)
            queue = top_level_keys.copy()
            min_key = None
            min_distance = 4800

            ########## Iterate trough top_level_keys ##########
            while True:
                distance = 4800

                ########## Iterate trough indexes of key ##########
                input_overlap_value = True if len(dataset[keys_and_durations[i][1]]['arousal']) >= dataset[keys_and_durations[i][1]]['duration'] / 5 else False
                dataset_overlap_value = True if len(dataset[queue[0]]['arousal']) >= dataset[queue[0]]['duration'] / 5 else False
                loop_correction = -1 if not input_overlap_value and dataset_overlap_value and dataset[queue[0]]['duration']%5 < dataset[keys_and_durations[i][1]]['duration'] %5 else 0
                if (input_overlap_value and not dataset_overlap_value and dataset[queue[0]]['duration']%5 < dataset[keys_and_durations[i][1]]['duration'] %5) or (input_overlap_value and dataset_overlap_value and dataset[queue[0]]['duration']%5 > dataset[keys_and_durations[i][1]]['duration'] %5):
                    loop_correction = 1
                elif input_overlap_value and not dataset_overlap_value and dataset[queue[0]]['duration']%5 > dataset[keys_and_durations[i][1]]['duration'] %5:
                    loop_correction = 2

                for j in range(0, int(len(dataset[queue[0]]['arousal']) - len(dataset[keys_and_durations[i][1]]['arousal'])) + loop_correction):
                    distance1 = 0

                    last_input_segment_value_correction = 1 if input_overlap_value and not dataset_overlap_value and dataset[queue[0]]['duration']%5 >= dataset[keys_and_durations[i][1]]['duration'] %5 else 0

                    ########## Get distance of index ##########
                    for k in range(0, len(dataset[keys_and_durations[i][1]]['arousal']) - last_input_segment_value_correction):
                        distance1 += (writing_weight_1 * abs(dataset[queue[0]]['arousal'][j+k] - dataset[keys_and_durations[i][1]]['arousal'][k])) ** exponent
                        distance1 += (writing_weight_2 * abs(dataset[queue[0]]['valence'][j+k] - dataset[keys_and_durations[i][1]]['valence'][k])) ** exponent

                    avg_distance = distance1 / (2 * (len(dataset[keys_and_durations[i][1]]['arousal']) - last_input_segment_value_correction))

                    if avg_distance < distance:
                        distance = avg_distance

                if distance < min_distance:
                    min_distance = distance
                    min_key = queue[0]
                queue.pop(0)

                if queue == []:
                    if ordering[min_key] != [] and math.ceil(dataset[ordering[min_key][0]]['duration']/5) > math.ceil(dataset[keys_and_durations[i][1]]['duration']/5):
                        queue += ordering[min_key]
                        min_key = None
                        min_distance = 4800
                    else:
                        ordering[min_key].append(keys_and_durations[i][1])
                        break

        offline_values = "static/datasets/audio/offline_values" + text_file_ending
        f = open(offline_values, "w")
        f.write(str(top_level_keys)+"\n")
        values = json.dumps(ordering)
        f.write(values)
        f.close()
        ########## Finished creating ordering ##########

        ########## Check if all 'parent keys' have more values ##########
        for item in keys_and_durations:
            key = item[1]
            for i in range(0, len(ordering[key])):
                print(math.ceil(dataset[key]['duration']/5) > dataset[ordering[key][i]]['duration']/5)

    for file in files_to_delete:
        if os.path.isdir(file):
            shutil.rmtree(file, ignore_errors=True)
        elif os.path.isfile(file):
            os.remove(file)

else:
    print("The audio_folder_path is incorrect. Please check.")
