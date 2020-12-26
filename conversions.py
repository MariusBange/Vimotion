'''different conversions and extractions'''
import csv
import cv2
import ffmpeg
import math
from mutagen.mp3 import MP3
import os
import subprocess
import time
from schedule import set_delete_timer

def convert_to_mp4(video):
    '''converting video to .mp4 using ffmpeg'''
    input_video = ffmpeg.input('static/uploads/video/' + video)
    dot_index = video.rfind('.')
    raw_filename = video[0:dot_index]
    filename = raw_filename
    i = 1

    while os.path.isfile('static/uploads/video/' + filename + '.mp4'):
        i += 1
        filename = raw_filename + '(' + str(i) + ')'

    ffmpeg.output(input_video, 'static/uploads/video/' + filename + '.mp4').run()
    set_delete_timer('static/uploads/video/' + filename + '.mp4')

def convert_to_mp3(audio):
    '''converting audio to .mp3 using ffmpeg'''
    input_audio = ffmpeg.input('static/uploads/audio/' + audio)
    dot_index = audio.rfind('.')
    raw_filename = audio[0:dot_index]
    filename = raw_filename
    i = 1

    while os.path.isfile('static/uploads/audio/' + filename + '.mp3'):
        i += 1
        filename = raw_filename + '(' + str(i) + ')'

    ffmpeg.output(input_audio, 'static/uploads/audio/' + filename + '.mp3').run()
    schedule.set_delete_timer('static/uploads/audio/' + filename + '.mp3')

def get_frames(video):
    '''extracting frames from input video using OpenCV

    depending on the frame rate, trying to extract 8 frames per second and store them
    in a folder together with a .csv file containing the frame names
    '''
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = float(result.stdout)

    cap = cv2.VideoCapture(video)
    slash_index = video.rfind('/')
    video = video[slash_index:len(video)+1]
    dot_index = video.rfind('.')
    video = video[0:dot_index]
    folder = video + '-frames(1)'
    i = 1

    while os.path.exists('static/frames'+folder):
        i += 1
        folder = video + '-frames(' + str(i) + ')'
    os.makedirs('static/frames'+folder)
    set_delete_timer('static/frames'+folder)

    frame_rate = cap.get(5)
    frames_per_segment = 5 * frame_rate
    print("Extracting frames of clip", video, "with duration:", duration, "and FPS:", frame_rate)

    if frames_per_segment < 64:
        return "Error", "Error"

    f = open('static/frames' + folder + '/values.txt', "a")
    f.write(str(duration))
    f.close()

    frame_number = 1
    frames = [[]]
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        frame_id = cap.get(1)
        frame_name = 'static/frames' + folder + '/' + 'frame' + str(frame_number) +  '.jpg'

        if frame is not None:
            # print ('Creating...' + frame_name)
            segment = int(frame_number // frames_per_segment)

            if len(frames) < segment + 1:
                frames.append([])

            frames[segment].append('frame' + str(frame_number) +  '.jpg')
            cv2.imwrite(frame_name, frame)
            frame_number += 1
        else:
            print ('Extracted', frame_number-1, 'frames in', time.time() - start_time, 'seconds')
            break

    with open('static/frames' + folder + '/' + 'frames.csv', 'w', newline='') as f:
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

        with open('static/frames' + folder + '/' + 'extension.csv', 'w', newline='') as f:
            writer = csv.writer(f)

            for i in range(0, last_segment_size):
                writer.writerow([frames[-1][i]])

    cap.release()
    cv2.destroyAllWindows()
    return folder, last_segment_size

def get_audio_clips(audio):
    '''splitting an audio file into segments of 5 seconds'''
    mp3_file = MP3(audio)
    duration = mp3_file.info.length

    if duration < 5:
        return "Error"
    else:
        audio_string = audio
        slash_index = audio_string.rfind('/')
        audio_string = audio_string[slash_index:len(audio_string)+1]
        dot_index = audio_string.rfind('.')
        audio_string = audio_string[0:dot_index]
        folder = audio_string + '-clips(1)'
        i = 1

        while os.path.exists('static/audioClips'+folder):
            i += 1
            folder = audio_string + '-clips(' + str(i) + ')'

        os.makedirs('static/audioClips'+folder+'/audio')
        set_delete_timer('static/audioClips'+folder)

        f = open("static/audioClips"+folder+"/values.txt", "w")
        f.write(str(duration))
        f.close()

        wav_file = 'static/uploads/audio/' + audio_string + '.wav'
        subprocess.call(['ffmpeg', '-i', audio, wav_file])
        set_delete_timer('static/uploads/audio/' + audio_string + '.wav')
        segments = math.ceil(duration / 5)

        for i in range(0, segments):
            start_time = str(i * 5) + '.0'
            if i < segments - 1:
                end_time = '5.0'
            else:
                end_time = str(duration % 5)
            if float(end_time) > 1:
                subprocess.call(['ffmpeg', '-ss', start_time, '-i', wav_file, '-c', 'copy', '-t', end_time, 'static/audioClips' + folder + '/audio/' + audio_string + '(' + str(i) + ')' + '.wav'])
        return folder
