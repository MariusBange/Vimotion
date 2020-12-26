'''different options to find matching audio and video'''
import json
import math
import time
import random
from ast import literal_eval

def get_input(folder, fileformat):
    '''get data from input file'''

    if fileformat == "video":
        values = 'static/frames' + folder + "/values.txt"
    elif fileformat == "audio":
        values = 'static/audioClips' + folder + "/values.txt"

    with open(values) as f:
        duration = float(f.readline().strip())
        arousal_string = f.readline().strip()
        valence_string = f.readline().strip()

    arousal = [float(value.strip()) for value in arousal_string.split(', ')]
    valence = [float(value.strip()) for value in valence_string.split(', ')]

    return duration, arousal, valence

def get_audio_dataset():
    '''get data from audio data set files'''
    values = "static/datasets/audio/dataset_values.txt"
    # values = "/Users/marius/OneDrive/Uni/Bachelor/Bachelorarbeit/Vimotions/Testing/dataset_values.txt"

    with open(values) as f:
        dataset = json.load(f)
    f.close()

    return dataset

def get_video_dataset():
    '''get data from video data set files'''
    values = "static/datasets/video/dataset_values.txt"
    # values = "/Users/marius/OneDrive/Uni/Bachelor/Bachelorarbeit/Vimotions/Testing/dataset_values.txt"

    with open(values) as f:
        dataset = json.load(f)
    f.close()

    return dataset


def brute_force_matching(folder, fileformat, mode):
    '''iterating through all combinations of data set items to find the best possible matching'''
    start_time = time.time()
    duration, arousal, valence = get_input(folder, fileformat)
    weight_1 = 1 if mode == "standard" else 1.9
    weight_2 = 1 if mode == "standard" else 0.1
    exponent = 2 if mode == "mse" else 1

    datset = {}
    longer_keys = []
    shorter_keys = []
    min_distance = float("inf")
    min_keys = []
    min_indexes = []


    if fileformat == "audio":
        dataset = get_video_dataset()

    elif fileformat == "video":
        dataset = get_audio_dataset()

    for file1 in dataset:
        keys_1 = []
        indexes_1 = []
        start_index_2 = 0
        longer_1 = dataset[file1]['duration'] >= duration and len(dataset[file1]['arousal']) >= len(arousal) - 1
        distance1 = 0

        if longer_1:
            correction11 = 0 if dataset[file1]['duration'] // 5 == duration // 5 and len(dataset[file1]['arousal']) > len(arousal) else 1
            correction12 = 1 if dataset[file1]['duration'] // 5 == duration // 5 and len(dataset[file1]['arousal']) < len(arousal) else 0

            for i in range(0, len(dataset[file1]['arousal']) - len(arousal) + correction11):
                distance1 = 0

                for j in range(0, len(arousal) - correction12):
                    distance1 += (weight_1 * abs(dataset[file1]['arousal'][i+j] - arousal[j])) ** exponent
                    distance1 += (weight_2 * abs(dataset[file1]['valence'][i+j] - valence[j])) ** exponent

                if distance1 < min_distance:
                    min_distance = distance1
                    min_keys = [file1]
                    min_indexes = [[i, i + math.ceil(duration/5)]]

            continue

        else:
            correction11 = 1 if dataset[file1]['duration'] // 5 == duration // 5 and len(dataset[file1]['arousal']) <= len(arousal) else 0

            for i in range(0, len(dataset[file1]['arousal']) - correction11):
                distance1 += (weight_1 * abs(dataset[file1]['arousal'][i] - arousal[i])) ** exponent
                distance1 += (weight_2 * abs(dataset[file1]['valence'][i] - valence[i])) ** exponent

            keys_1 = [file1]
            key_correction_1 = 2 if dataset[file1]['duration'] // 5 == duration // 5 and ((len(dataset[file1]['arousal']) == len(arousal) and duration // 5 == len(arousal)) or len(dataset[file1]['arousal']) > len(arousal)) else 1
            indexes_1 = [[0, int(dataset[file1]['duration'] // 5 - key_correction_1)]]
            start_index_2 = int(dataset[file1]['duration'] // 5 - key_correction_1 + 1)


        for file2 in dataset:
            keys_2 = keys_1.copy()
            indexes_2 = indexes_1.copy()
            start_index_3 = 0
            distance2 = distance1
            keys_to_transfer_2 = []
            longer_2 = dataset[file2]['duration'] >= duration - start_index_2 * 5 and len(dataset[file2]['arousal']) >= len(arousal) - 1 - start_index_2

            if longer_2:
                correction21 = 0 if dataset[file2]['duration'] // 5 == duration // 5 - start_index_2 and len(dataset[file2]['arousal']) > len(arousal) - start_index_2 else 1
                correction22 = 1 if dataset[file2]['duration'] // 5 == duration // 5 - start_index_2 and len(dataset[file2]['arousal']) < len(arousal) - start_index_2 else 0

                for i in range(0, len(dataset[file2]['arousal']) - len(arousal) + start_index_2 + correction21):
                    distance2 = distance1

                    for j in range(start_index_2, len(arousal) - correction22):
                        distance2 += (weight_1 * abs(dataset[file2]['arousal'][i+j-start_index_2] - arousal[j])) ** exponent
                        distance2 += (weight_2 * abs(dataset[file2]['valence'][i+j-start_index_2] - valence[j])) ** exponent

                    if distance2 < min_distance:
                        min_distance = distance2
                        min_keys = keys_1 + [file2]
                        min_indexes = indexes_1 + [[i, int(i+math.ceil(duration/5-start_index_2))]]

                continue

            else:
                correction21 = 1 if dataset[file2]['duration'] // 5 == duration // 5 - start_index_2 and len(dataset[file2]['arousal']) <= len(arousal)- start_index_2 else 0

                for i in range(0, len(dataset[file2]['arousal']) - correction21):
                    distance2 += (weight_1 * abs(dataset[file2]['arousal'][i] - arousal[i+start_index_2])) ** exponent
                    distance2 += (weight_2 * abs(dataset[file2]['valence'][i] - valence[i+start_index_2])) ** exponent

                keys_2.append(file2)
                key_correction_2 = 2 if dataset[file2]['duration'] // 5 == duration // 5 - start_index_2 and ((len(dataset[file2]['arousal']) == len(arousal) - start_index_2 and duration // 5 == len(arousal)) or len(dataset[file2]['arousal']) > len(arousal) - start_index_2) else 1
                indexes_2.append([0, int(dataset[file2]['duration'] // 5 - key_correction_2)])
                start_index_3 = int(dataset[file2]['duration'] // 5 - key_correction_2 + 1 + start_index_2)

            for file3 in dataset:
                keys_3 = keys_2.copy()
                indexes_3 = indexes_2.copy()
                start_index_4 = 0
                distance3 = distance2
                keys_to_transfer_3 = []
                longer_3 = dataset[file3]['duration'] >= duration - start_index_3 * 5 and len(dataset[file3]['arousal']) >= len(arousal) - 1 - start_index_3

                if longer_3:
                    correction31 = 0 if dataset[file3]['duration'] // 5 == duration // 5 - start_index_3 and len(dataset[file3]['arousal']) > len(arousal) - start_index_3 else 1
                    correction32 = 1 if dataset[file3]['duration'] // 5 == duration // 5 - start_index_3 and len(dataset[file3]['arousal']) < len(arousal) - start_index_3 else 0

                    for i in range(0, len(dataset[file3]['arousal']) - len(arousal) + start_index_3  + correction31):
                        distance3 = distance2

                        for j in range(start_index_3, len(arousal) - correction32):
                            distance3 += (weight_1 * abs(dataset[file3]['arousal'][i+j-start_index_3] - arousal[j])) ** exponent
                            distance3 += (weight_2 * abs(dataset[file3]['valence'][i+j-start_index_3] - valence[j])) ** exponent

                        if distance3 < min_distance:
                            min_distance = distance3
                            min_keys = keys_2 + [file3]
                            min_indexes = indexes_2 + [[i, int(i+math.ceil(duration/5-start_index_3))]]

                    continue

                else:
                    correction31 = 1 if dataset[file3]['duration'] // 5 == duration // 5 - start_index_3 and len(dataset[file3]['arousal']) <= len(arousal)- start_index_3 else 0

                    for i in range(0, len(dataset[file3]['arousal']) - correction31):
                        distance3 += (weight_1 * abs(dataset[file3]['arousal'][i] - arousal[i+start_index_3])) ** exponent
                        distance3 += (weight_2 * abs(dataset[file3]['valence'][i] - valence[i+start_index_3])) ** exponent

                    keys_3.append(file3)
                    key_correction_3 = 2 if dataset[file3]['duration'] // 5 == duration // 5 - start_index_3 and ((len(dataset[file3]['arousal']) == len(arousal) - start_index_3 and duration // 5 == len(arousal)) or len(dataset[file3]['arousal']) > len(arousal) - start_index_3) else 1
                    indexes_3.append([0, int(dataset[file3]['duration'] // 5 - key_correction_3)])
                    start_index_4 = int(dataset[file3]['duration'] // 5 - key_correction_3 + 1 + start_index_3)

                for file4 in dataset:
                    keys_4 = keys_3.copy()
                    indexes_4 = indexes_3.copy()
                    start_index_5 = 0
                    distance4 = distance3
                    keys_to_transfer_4 = []
                    longer_4 = dataset[file4]['duration'] >= duration - start_index_4 * 5 and len(dataset[file4]['arousal']) >= len(arousal) - 1 - start_index_4

                    if longer_4:
                        correction41 = 0 if dataset[file4]['duration'] // 5 == duration // 5 - start_index_4 and len(dataset[file4]['arousal']) > len(arousal) - start_index_4 else 1
                        correction42 = 1 if dataset[file4]['duration'] // 5 == duration // 5 - start_index_4 and len(dataset[file4]['arousal']) < len(arousal) - start_index_4 else 0

                        for i in range(0, len(dataset[file4]['arousal']) - len(arousal) + start_index_4 + correction41):
                            distance4 = distance3

                            for j in range(start_index_4, len(arousal) - correction42):
                                distance4 += (weight_1 * abs(dataset[file4]['arousal'][i+j-start_index_4] - arousal[j])) ** exponent
                                distance4 += (weight_2 * abs(dataset[file4]['valence'][i+j-start_index_4] - valence[j])) ** exponent

                            if distance4 < min_distance:
                                min_distance = distance4
                                min_keys = keys_3 + [file4]
                                min_indexes = indexes_3 + [[i, int(i+math.ceil(duration/5-start_index_4))]]

                        continue

                    else:
                        correction41 = 1 if dataset[file4]['duration'] // 5 == duration // 5 - start_index_4 and len(dataset[file4]['arousal']) <= len(arousal)- start_index_4 else 0

                        for i in range(0, len(dataset[file4]['arousal']) - correction41):
                            distance4 += (weight_1 * abs(dataset[file4]['arousal'][i] - arousal[i+start_index_4])) ** exponent
                            distance4 += (weight_2 * abs(dataset[file4]['valence'][i] - valence[i+start_index_4])) ** exponent

                        keys_4.append(file4)
                        key_correction_4 = 2 if dataset[file4]['duration'] // 5 == duration // 5 - start_index_4 and ((len(dataset[file4]['arousal']) == len(arousal) - start_index_4 and duration // 5 == len(arousal)) or len(dataset[file4]['arousal']) > len(arousal) - start_index_4) else 1
                        indexes_4.append([0, int(dataset[file4]['duration'] // 5 - key_correction_4)])
                        start_index_5 = int(dataset[file4]['duration'] // 5 - key_correction_4 + 1 + start_index_4)

                    for file5 in dataset:
                        keys_5 = keys_4.copy()
                        indexes_5 = indexes_4.copy()
                        start_index_6 = 0
                        distance5 = distance4
                        keys_to_transfer_5 = []
                        longer_5 = dataset[file5]['duration'] >= duration - start_index_5 * 5 and len(dataset[file5]['arousal']) >= len(arousal) - 1 - start_index_5

                        if longer_5:
                            correction51 = 0 if dataset[file5]['duration'] // 5 == duration // 5 - start_index_5 and len(dataset[file5]['arousal']) > len(arousal) - start_index_5 else 1
                            correction52 = 1 if dataset[file5]['duration'] // 5 == duration // 5 - start_index_5 and len(dataset[file5]['arousal']) < len(arousal) - start_index_5 else 0

                            for i in range(0, len(dataset[file5]['arousal']) - len(arousal) + start_index_5  + correction51):
                                distance5 = distance4

                                for j in range(start_index_5, len(arousal) - correction52):
                                    distance5 += (weight_1 * abs(dataset[file5]['arousal'][i+j-start_index_5] - arousal[j])) ** exponent
                                    distance5 += (weight_2 * abs(dataset[file5]['valence'][i+j-start_index_5] - valence[j])) ** exponent

                                if distance5 < min_distance:
                                    min_distance = distance5
                                    min_keys = keys_4 + [file5]
                                    min_indexes = indexes_4 + [[i, int(i+math.ceil(duration/5-start_index_5))]]

                            continue

                        else:
                            correction51 = 1 if dataset[file5]['duration'] // 5 == duration // 5 - start_index_5 and len(dataset[file5]['arousal']) <= len(arousal)- start_index_5 else 0

                            for i in range(0, len(dataset[file5]['arousal']) - correction51):
                                distance5 += (weight_1 * abs(dataset[file5]['arousal'][i] - arousal[i+start_index_5])) ** exponent
                                distance5 += (weight_2 * abs(dataset[file5]['valence'][i] - valence[i+start_index_5])) ** exponent

                            keys_5.append(file5)
                            key_correction_5 = 2 if dataset[file5]['duration'] // 5 == duration // 5 - start_index_5 and ((len(dataset[file5]['arousal']) == len(arousal) - start_index_5 and duration // 5 == len(arousal)) or len(dataset[file5]['arousal']) > len(arousal) - start_index_5) else 1
                            indexes_5.append([0, int(dataset[file5]['duration'] // 5 - key_correction_5)])
                            start_index_6 = int(dataset[file5]['duration'] // 5 - key_correction_5 + 1 + start_index_5)

                        for file6 in dataset:
                            keys_6 = keys_5.copy()
                            indexes_6 = indexes_5.copy()
                            start_index_7 = 0
                            distance6 = distance5
                            keys_to_transfer_6 = []
                            longer_6 = dataset[file6]['duration'] >= duration - start_index_6 * 5 and len(dataset[file6]['arousal']) >= len(arousal) - 1 - start_index_6

                            if longer_6:
                                correction61 = 0 if dataset[file6]['duration'] // 5 == duration // 5 - start_index_6 and len(dataset[file6]['arousal']) > len(arousal) - start_index_6 else 1
                                correction62 = 1 if dataset[file6]['duration'] // 5 == duration // 5 - start_index_6 and len(dataset[file6]['arousal']) < len(arousal) - start_index_6 else 0

                                for i in range(0, len(dataset[file6]['arousal']) - len(arousal) + start_index_6 + correction61):
                                    distance6 = distance5

                                    for j in range(start_index_6, len(arousal) - correction62):
                                        distance6 += (weight_1 * abs(dataset[file6]['arousal'][i+j-start_index_6] - arousal[j])) ** exponent
                                        distance6 += (weight_2 * abs(dataset[file6]['valence'][i+j-start_index_6] - valence[j])) ** exponent

                                    if distance6 < min_distance:
                                        min_distance = distance6
                                        min_keys = keys_5 + [file6]
                                        min_indexes = indexes_5 + [[i, int(i+math.ceil(duration/5-start_index_6))]]

                                continue

                            else:
                                correction61 = 1 if dataset[file6]['duration'] // 5 == duration // 5 - start_index_6 and len(dataset[file6]['arousal']) <= len(arousal)- start_index_6 else 0

                                for i in range(0, len(dataset[file6]['arousal']) - correction61):
                                    distance6 += (weight_1 * abs(dataset[file6]['arousal'][i] - arousal[i+start_index_6])) ** exponent
                                    distance6 += (weight_2 * abs(dataset[file6]['valence'][i] - valence[i+start_index_6])) ** exponent

                                keys_6.append(file6)
                                key_correction_6 = 2 if dataset[file6]['duration'] // 5 == duration // 5 - start_index_6 and ((len(dataset[file6]['arousal']) == len(arousal) - start_index_6 and duration // 5 == len(arousal)) or len(dataset[file6]['arousal']) > len(arousal) - start_index_6) else 1
                                indexes_6.append([0, int(dataset[file6]['duration'] // 5 - key_correction_6)])
                                start_index_7 = int(dataset[file6]['duration'] // 5 - key_correction_6 + 1 + start_index_6)

                            for file7 in dataset:
                                keys_7 = keys_6.copy()
                                indexes_7 = indexes_6.copy()
                                start_index_8 = 0
                                distance7 = distance6
                                keys_to_transfer_7 = []
                                longer_7 = dataset[file7]['duration'] >= duration - start_index_7 * 5 and len(dataset[file7]['arousal']) >= len(arousal) - 1 - start_index_7

                                if longer_7:
                                    correction71 = 0 if dataset[file7]['duration'] // 5 == duration // 5 - start_index_7 and len(dataset[file7]['arousal']) > len(arousal) - start_index_7 else 1
                                    correction72 = 1 if dataset[file7]['duration'] // 5 == duration // 5 - start_index_7 and len(dataset[file7]['arousal']) < len(arousal) - start_index_7 else 0

                                    for i in range(0, len(dataset[file7]['arousal']) - len(arousal) + start_index_7 + correction71):
                                        distance7 = distance6

                                        for j in range(start_index_7, len(arousal) - correction72):
                                            distance7 += (weight_1 * abs(dataset[file7]['arousal'][i+j-start_index_7] - arousal[j])) ** exponent
                                            distance7 += (weight_2 * abs(dataset[file7]['valence'][i+j-start_index_7] - valence[j])) ** exponent

                                        if distance7 < min_distance:
                                            min_distance = distance7
                                            min_keys = keys_6 + [file7]
                                            min_indexes = indexes_6 + [[i, int(i+math.ceil(duration/5-start_index_7))]]

                                    continue

                                else:
                                    correction71 = 1 if dataset[file7]['duration'] // 5 == duration // 5 - start_index_7 and len(dataset[file7]['arousal']) <= len(arousal)- start_index_7 else 0

                                    for i in range(0, len(dataset[file7]['arousal']) - correction71):
                                        distance7 += (weight_1 * abs(dataset[file7]['arousal'][i] - arousal[i+start_index_7])) ** exponent
                                        distance7 += (weight_2 * abs(dataset[file7]['valence'][i] - valence[i+start_index_7])) ** exponent

                                    keys_7.append(file7)
                                    key_correction_7 = 2 if dataset[file7]['duration'] // 5 == duration // 5 - start_index_7 and ((len(dataset[file7]['arousal']) == len(arousal) - start_index_7 and duration // 5 == len(arousal)) or len(dataset[file7]['arousal']) > len(arousal) - start_index_7) else 1
                                    indexes_7.append([0, int(dataset[file7]['duration'] // 5 - key_correction_7)])
                                    start_index_8 = int(dataset[file7]['duration'] // 5 - key_correction_7 + 1 + start_index_7)

                                for file8 in dataset:
                                    keys_8 = keys_7.copy()
                                    indexes_8 = indexes_7.copy()
                                    distance8 = distance7

                                    correction81 = 0 if dataset[file8]['duration'] // 5 == duration // 5 - start_index_8 and len(dataset[file8]['arousal']) > len(arousal) - start_index_8 else 1
                                    correction82 = 1 if dataset[file8]['duration'] // 5 == duration // 5 - start_index_8 and len(dataset[file8]['arousal']) < len(arousal) - start_index_8 else 0

                                    for i in range(0, len(dataset[file8]['arousal']) - len(arousal) + start_index_8  + correction81):
                                        distance8 = distance7

                                        for j in range(start_index_8, len(arousal) - correction82):
                                            distance8 += (weight_1 * abs(dataset[file8]['arousal'][i+j-start_index_8] - arousal[j])) ** exponent
                                            distance8 += (weight_2 * abs(dataset[file8]['valence'][i+j-start_index_8] - valence[j])) ** exponent

                                        if distance8 < min_distance:
                                            min_distance = distance8
                                            min_keys = keys_7 + [file8]
                                            min_indexes = indexes_7 + [[i, int(i+math.ceil(duration/5-start_index_8))]]

    ########## Return items with shortest (average) distances ##########

    return min_keys, min_indexes


def approximation_matching(folder, fileformat, mode):
    '''iterating through the length of input to find the best possible matching'''
    start_time = time.time()
    duration, arousal, valence = get_input(folder, fileformat)
    weight_1 = 1 if mode == "standard" else 1.9
    weight_2 = 1 if mode == "standard" else 0.1
    exponent = 2 if mode == "mse" else 1

    datset = {}
    min_keys = []
    min_avg_distances = []
    min_indexes = []
    matching_duration = 0

    if fileformat == "audio":
        dataset = get_video_dataset()

    elif fileformat == "video":
        dataset = get_audio_dataset()

    shorter_keys = list(dataset.keys())

    for x in range(0, 8):
        min_avg_distances.append(float("inf"))
        min_keys.append("")
        min_indexes.append([0, 0])
        keys_to_pop = []

        for file in dataset:
            longer = dataset[file]['duration'] >= duration - matching_duration * 5 and len(dataset[file]['arousal']) >= len(arousal) - 1 - matching_duration

            if longer:
                correction1 = 0 if dataset[file]['duration'] // 5 == duration // 5 - matching_duration and len(dataset[file]['arousal']) > len(arousal) - matching_duration else 1
                correction2 = 1 if dataset[file]['duration'] // 5 == duration // 5 - matching_duration and len(dataset[file]['arousal']) < len(arousal) - matching_duration else 0

                for i in range(0, len(dataset[file]['arousal']) - len(arousal) + matching_duration + correction1):
                    distance = 0

                    for j in range(matching_duration, len(arousal) - correction2):
                        distance += (weight_1 * abs(dataset[file]['arousal'][i+j-matching_duration] - arousal[j])) ** exponent
                        distance += (weight_2 * abs(dataset[file]['valence'][i+j-matching_duration] - valence[j])) ** exponent

                    avg_distance = distance / (2 * (len(arousal) - correction2 - matching_duration))

                    if avg_distance < min_avg_distances[x]:
                        min_avg_distances[x] = avg_distance
                        min_keys[x] = file
                        min_indexes[x] = [i, int(i+math.ceil(duration/5-matching_duration))]
            else:
                correction3 = 1 if dataset[file]['duration'] // 5 == duration // 5 - matching_duration and len(dataset[file]['arousal']) <= len(arousal) - matching_duration else 0
                distance = 0

                for i in range(0, len(dataset[file]['arousal']) - correction3):
                    distance += (weight_1 * abs(dataset[file]['arousal'][i] - arousal[i+matching_duration])) ** exponent
                    distance += (weight_2 * abs(dataset[file]['valence'][i] - valence[i+matching_duration])) ** exponent

                avg_distance = distance / (2 * len(dataset[file]['arousal']) - correction3)

                if avg_distance < min_avg_distances[x]:
                    min_avg_distances[x] = avg_distance
                    min_keys[x] = file
                    key_correction = 2 if dataset[file]['duration'] // 5 == duration // 5 - matching_duration and ((len(dataset[file]['arousal']) == len(arousal) - matching_duration and duration // 5 == len(arousal)) or len(dataset[file]['arousal']) > len(arousal) - matching_duration) else 1
                    min_indexes[x] = [0, int(dataset[file]['duration']//5-key_correction)]

        matching_duration += min_indexes[x][1] - min_indexes[x][0] + 1
        if matching_duration * 5 > duration:
            break

    ########## Return items with shortest (average) distances ##########

    return min_keys, min_indexes


def offline_matching(folder, fileformat, mode):
    '''iterating through all data set items to find the best possible matching'''
    start_time = time.time()
    duration, arousal, valence = get_input(folder, fileformat)
    weight_1 = 1 if mode == "standard" else 1.9
    weight_2 = 1 if mode == "standard" else 0.1
    exponent = 2 if mode == "mse" else 1
    text_file_ending = ".txt" if mode == "standard" else "_weighted.txt"

    if mode == "mse":
        text_file_ending = "_mse.txt"

    if fileformat == "audio":
        dataset = get_video_dataset()

    elif fileformat == "video":
        dataset = get_audio_dataset()

    ########## Start actual matching ##########
    if fileformat == "audio":
        offline_values = "static/datasets/video/offline_values" + text_file_ending
    else:
        offline_values = "static/datasets/audio/offline_values" + text_file_ending
    f = open(offline_values, "r")
    lines = f.readlines()
    f.close()

    top_level_keys = literal_eval(lines[0])
    ordering = json.loads(lines[1])

    queue = top_level_keys.copy()
    best_longer_key = None
    longer_index = None
    best_longer_distance = float("inf")
    temp_key = None
    temp_index = None
    temp_distance = float("inf")

    ########## Iterate trough keys with longer duration ##########
    while True:
        distance = float("inf")
        index = 0

        ########## Iterate trough indexes of key ##########
        input_overlap_value = True if len(arousal) > duration / 5 else False
        dataset_overlap_value = True if len(dataset[queue[0]]['arousal']) > dataset[queue[0]]['duration'] / 5 else False

        if dataset[queue[0]]['duration'] >= duration:

            loop_correction = -1 if not input_overlap_value and dataset_overlap_value and dataset[queue[0]]['duration']%5 < duration %5 else 0
            if (input_overlap_value and not dataset_overlap_value and dataset[queue[0]]['duration']%5 < duration %5) or (input_overlap_value and dataset_overlap_value and dataset[queue[0]]['duration']%5 > duration%5):
                loop_correction = 1
            elif input_overlap_value and not dataset_overlap_value and dataset[queue[0]]['duration']%5 > duration %5:
                loop_correction = 2

            for i in range(0, int(len(dataset[queue[0]]['arousal']) - len(arousal)) + loop_correction):
                distance1 = 0

                last_input_segment_value_correction = 1 if input_overlap_value and not dataset_overlap_value and dataset[queue[0]]['duration']%5 >= duration %5 else 0

                ########## Get distance of index ##########
                for j in range(0, len(arousal) - last_input_segment_value_correction):
                    distance1 += (weight_1 * abs(dataset[queue[0]]['arousal'][i+j] - arousal[j])) ** exponent
                    distance1 += (weight_2 * abs(dataset[queue[0]]['valence'][i+j] - valence[j])) ** exponent

                avg_distance = distance1 / (2 * (len(arousal) - last_input_segment_value_correction))

                if avg_distance < distance:
                    distance = avg_distance
                    index = i

            if distance < best_longer_distance:
                best_longer_distance = distance
                best_longer_key = queue[0]
                longer_index = [index, index + int(duration//5)]
                temp_distance = distance
                temp_index = [index, index + int(duration//5)]
                temp_key = queue[0]
            elif distance < temp_distance:
                temp_distance = distance
                temp_index = [index, index + int(duration//5)]
                temp_key = queue[0]
            queue.pop(0)

            if queue == []:
                if ordering[temp_key] != [] and math.ceil(dataset[ordering[temp_key][0]]['duration']/5) > math.ceil(duration/5):
                    queue += ordering[temp_key]
                    temp_key = None
                    temp_distance = float("inf")
                    temp_index = None
                else:
                    break
        else:
            queue.pop()
            if queue == []:
                if ordering[temp_key] != [] and math.ceil(dataset[ordering[temp_key][0]]['duration']/5) > math.ceil(duration/5):
                    queue += ordering[temp_key]
                    temp_key = None
                    temp_distance = float("inf")
                    temp_index = None
                else:
                    break

    ########## Return longer data set item with shortest (average) distances ##########

    return [best_longer_key], [longer_index]


def random_matching(folder, fileformat):
    '''iterating through all data set items to find the best possible matching'''
    start_time = time.time()
    duration, arousal, valence = get_input(folder, fileformat)

    if fileformat == "audio":
        dataset = get_video_dataset()

    elif fileformat == "video":
        dataset = get_audio_dataset()

    min_keys = []
    min_indexes = []
    matching_duration = 0

    while True:

        min_keys.append(random.choice(list(dataset.keys())))

        while dataset[min_keys[-1]]['duration'] < duration - matching_duration * 5 and len(dataset[min_keys[-1]]['arousal']) == len(arousal) - matching_duration:
            min_keys[-1] = random.choice(list(dataset.keys()))

        if dataset[min_keys[-1]]['duration'] >= duration - matching_duration * 5:
            start_index = random.randint(0, dataset[min_keys[-1]]['duration']//5 - (duration//5 - matching_duration))
            min_indexes.append([start_index, int(start_index+math.ceil(duration/5-matching_duration))])
            break
        else:
            min_indexes.append([0, min(len(arousal) - 2, len(dataset[min_keys[-1]]['arousal']) - 1)])
            matching_duration += min_indexes[-1][1] + 1

    return min_keys, min_indexes
