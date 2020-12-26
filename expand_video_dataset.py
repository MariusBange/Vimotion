'''add video files to audio data set'''
import os
import ffmpeg
import subprocess
import shutil
import math
import json
import csv
import cv2
import time

video_folder_path = "" #Path of folder of the video file(s) which should be added. IMPORTANT: Only audio files in this folder which should be added to the dataset!

if os.path.isdir(video_folder_path):

    dataset = {}
    errors = []
    dataset_values = "static/datasets/video/dataset_values.txt"

    with open(dataset_values) as f:
        dataset = json.load(f)
    f.close()

    files = [f for f in os.listdir(video_folder_path) if os.path.isfile(os.path.join(video_folder_path, f))]

    mp4_files = []
    folders = []
    files_to_delete = []
    copied_files = []

    ########## Convert files to .mp4 ##########
    for file in files:
        if file[-4:] != '.mp4':
            if file[-4:] not in ['.avi', '.flv', '.mov', '.mpeg', '.wmv']:
                continue
            input_video = ffmpeg.input(os.path.join(video_folder_path, file))
            dot_index = video.rfind('.')
            raw_filename = video[0:dot_index]
            filename = raw_filename
            i = 1

            while os.path.isfile(os.path.join(video_folder_path, filename) + '.mp4'):
                i += 1
                filename = raw_filename + '(' + str(i) + ')'

            ffmpeg.output(input_video, os.path.join(video_folder_path, filename) + '.mp4').run()
            mp4_files.append(filename + '.mp4')
        else:
            mp4_files.append(file)

    ########## Copy files into dataset ##########
    for file in mp4_files:
        video_string = file
        dot_index = video_string.rfind('.')
        video_string = video_string[0:dot_index]

        filename = video_string + '.mp4'
        i = 0
        while os.path.isfile('static/datasets/video/' + filename):
            i += 1
            filename = video_string + '(' + str(i) + ').mp4'
        shutil.copy(os.path.join(video_folder_path, file), 'static/datasets/video/' + filename)
        copied_files.append(filename)

    ########## Feature extraction for each file ##########
    for file in copied_files:
        duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", "static/datasets/video/" + filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout)

        if duration < 40:
            continue
        else:
            video_string = file
            dot_index = video_string.rfind('.')
            video_string = video_string[0:dot_index]
            folder = video_string + '-frames(1)'
            i = 1

            while os.path.exists('static/frames/'+folder):
                i += 1
                folder = video_string + '-frames(' + str(i) + ')'
            folders.append(folder)
            os.makedirs('static/frames/'+folder)

            ########## Extract frames ##########
            video = 'static/datasets/video/' + video_string + '.mp4'
            cap = cv2.VideoCapture(video)
            slash_index = video.rfind('/')
            video = video[slash_index:len(video)+1]
            dot_index = video.rfind('.')
            video = video[0:dot_index]

            frame_rate = cap.get(5)
            frames_per_segment = 5 * frame_rate

            if frames_per_segment < 64:
                errors.append(video_string)
                continue

            f = open("static/frames/"+folder+"/values.txt", "w")
            f.write(str(duration))
            f.close()

            frame_number = 1
            frames = [[]]
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                frame_id = cap.get(1)
                frame_name = 'static/frames/' + folder + '/' + 'frame' + str(frame_number) +  '.jpg'

                if frame is not None:
                    segment = int(frame_number // frames_per_segment)

                    if len(frames) < segment + 1:
                        frames.append([])

                    frames[segment].append('frame' + str(frame_number) +  '.jpg')
                    cv2.imwrite(frame_name, frame)
                    frame_number += 1
                else:
                    break

            with open('static/frames/' + folder + '/' + 'frames.csv', 'w', newline='') as f:
                writer = csv.writer(f)

                for segment in frames:
                    segment_size = len(segment)

                    if segment_size >= 64:
                        segment_median = segment_size // 2
                        segment_start = segment_median - 31

                        for i in range(segment_start, segment_start + 64):
                            writer.writerow([segment[i]])

            last_segment_size = len(frames[-1])

            if last_segment_size < 64 and last_segment_size >= 2:

                with open('static/frames/' + folder + '/' + 'extension.csv', 'w', newline='') as f:
                    writer = csv.writer(f)

                    for i in range(0, last_segment_size):
                        writer.writerow([frames[-1][i]])

            cap.release()
            cv2.destroyAllWindows()

            extension = last_segment_size

            ########## Extract features ##########
            if extension < 64 and extension >= 2:
                os.system("python3.7 Model/ResNet50/feature_extraction_ResNet50.py '/" + folder + "' '64'")
                os.system("python3.7 Model/I3D_RGB/feature_extraction_I3D_RGB.py '/" + folder + "' '64'")
                os.system("python3.7 Model/FlowNetS/feature_extraction_FlowNetS.py '/" + folder + "' '64'")
                os.system("python3.7 Model/AttendAffectNet/video_model_arousal.py '/" + folder + "' '64'")
                os.system("python3.7 Model/AttendAffectNet/video_model_valence.py '/" + folder + "' '64'")
                os.system("python3.7 Model/ResNet50/feature_extraction_resnet50.py '/" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/I3D_RGB/feature_extraction_I3D_RGB.py '/" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/FlowNetS/feature_extraction_FlowNetS.py '/" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/AttendAffectNet/video_model_arousal.py '/" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/AttendAffectNet/video_model_valence.py '/" + folder + "' '" + str(extension) + "'")
            else:
                os.system("python3.7 Model/ResNet50/feature_extraction_ResNet50.py '/" + folder + "' '64'")
                os.system("python3.7 Model/I3D_RGB/feature_extraction_I3D_RGB.py '/" + folder + "' '64'")
                os.system("python3.7 Model/FlowNetS/feature_extraction_FlowNetS.py '/" + folder + "' '64'")
                os.system("python3.7 Model/AttendAffectNet/video_model_arousal.py '/" + folder + "' '64'")
                os.system("python3 Model/AttendAffectNet/video_model_valence.py '/" + folder + "' '64'")

            values = 'static/frames/' + folder + "/values.txt"

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
                    errors.append(video_string)
                    break
            if video_string not in errors:
                dataset[video_string] = {'duration': duration, 'arousal': arousal,'valence': valence}
            else:
                print("ERROR: The file ", video_string, " could not be added due to an error while extracting features.")

    f = open(dataset_values, "w")
    dataset_updated = json.dumps(dataset)
    f.write(dataset_updated)
    f.close()

    ########## Add files to metadata.csv ##########
    metadata_rows = []

    with open('static/datasets/video/metadata.csv', newline='') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            metadata_rows.append(row)

    for file in copied_files:
        metadata_rows.append([file, "", ""])

    with open('static/datasets/video/metadata.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(metadata_rows)

    files_to_delete.append('static/frames/' + folder)

    modes = ["standard", "weighted", "mse"]

    for mode in modes:
        exponent = 2 if mode == "mse" else 1
        writing_weight_1 = 1 if mode == "standard" else 1.9
        writing_weight_2 = 1 if mode == "standard" else 0.1
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

        offline_values = "static/datasets/video/offline_values" + text_file_ending
        f = open(offline_values, "w")
        f.write(str(top_level_keys)+"\n")
        values = json.dumps(ordering)
        f.write(values)
        f.close()
        ########## Finished updating ordering ##########

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
    print("The video_folder_path is incorrect. Please check.")
