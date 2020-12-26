'''remove video files from audio data set'''
import os
import csv
import json
import math

files_to_remove = ["[HD] Pixar - For The Birds  Original Movie from Pixar"] #Name of all video file(s) which should be removed from the data set. IMPORTANT: List the filenames WITHOUT the file ending (i.e. "Mulan Avalanche Scene" instead of "Mulan Avalanche Scene.mp4")!

video_dataset_path = 'static/datasets/video'
files_with_ending = []
errors = []


########## Remove files from data set ##########
for file in files_to_remove:
    if os.path.isfile(os.path.join(video_dataset_path, file + '.mp4')):
        os.remove(os.path.join(video_dataset_path, file + '.mp4'))
        files_with_ending.append(file + '.mp4')
    else:
        errors.append(file)

########## Remove data of files from metadata.csv ##########
metadata_rows = []

with open('static/datasets/video/metadata.csv', newline='') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if row[0] not in files_with_ending:
            metadata_rows.append(row)

with open('static/datasets/video/metadata.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(metadata_rows)

########## Remove data of files from dataset_values.txt ##########
dataset = {}
dataset_values = "static/datasets/video/dataset_values.txt"

with open(dataset_values) as f:
    dataset = json.load(f)
f.close()

for file in files_with_ending:
    dataset.pop(file[:-4])

f = open(dataset_values, "w")
dataset_updated = json.dumps(dataset)
f.write(dataset_updated)
f.close()

########## Remove data of files from offline_values text files ##########
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
    ########## Finished creating ordering ##########

    ########## Check if all 'parent keys' have more values ##########
    for item in keys_and_durations:
        key = item[1]
        for i in range(0, len(ordering[key])):
            print(math.ceil(dataset[key]['duration']/5) > dataset[ordering[key][i]]['duration']/5)


if len(errors) > 0:
    print("The following files couldn't be found and therefore not removed: ", errors)
else:
    print("All files have beed removed")
