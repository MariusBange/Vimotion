import conversions #For converting files to .mp3 or .mp4 and for extracting frames or 5 second segments
import csv #For reading csv files
import ffmpeg #For operations with audio and video (like merging) (https://ffmpeg.org)
import os #For accessing paths and saving files
import pytube #For downloading videos from YouTube links (https://github.com/get-pytube/pytube3)
import subprocess #For calling terminal commands
from cameraAccess import VideoCamera #For accessing the users camera
from encryption import encode, decode #For security reasons
from flask import Flask, jsonify, redirect, render_template, request, Response, send_from_directory, session, url_for
from matching import brute_force_matching, approximation_matching, offline_matching, random_matching #For matching audio and video files
from mutagen.mp3 import MP3 #For getting audio file properties (https://pypi.org/project/mutagen/)
from pydub import AudioSegment #For merging multiple .mp3 files to one file (https://github.com/jiaaro/pydub/blob/master/API.markdown)
from pytube import YouTube #See line 5 import pytube, YouTube object: (https://python-pytube.readthedocs.io/en/latest/api.html#youtube-object)
from schedule import set_delete_timer #For deleting uploaded files after 24 hours
from videoprops import get_video_properties #For getting video file properties (https://pypi.org/project/get-video-properties/)

global_frame = None
video_camera = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bachelor 2020'


@app.before_request
def before_request():
    '''removing session item before visiting home page or camera page'''
    if request.path == "/":
        session.pop('files', None)
    elif request.path == "/camera":
        if 'files' in session and 'matching' in session['files']:
            session['files'] = {'matching': session['files']['matching']}
        else:
            session.pop('files', None)


@app.route("/")
def home():
    '''home route returning a simple html'''
    reset_camera()
    return render_template("home.html")


@app.route("/select_matching", methods=['GET', 'POST'])
def select_matching():
    if request.method == 'POST' and request.form.get('matching') != None:
        matching = request.form.get('matching')
        session['files'] = {'matching': matching}
        return render_template("select_matching.html", matching=matching)
    elif 'files' in session and 'matching' in session['files']:
        matching = session['files']['matching']
        return render_template("select_matching.html", matching=matching)
    if request.method == 'GET':
        session.pop('files', None)
    return render_template("select_matching.html", matching="")


@app.route("/audio", methods=['GET', 'POST'])
def song():
    '''route to upload an audio file'''
    if request.method == 'GET':
        if 'files' in session and 'matching' in session['files']:
            return render_template("audio.html", user_upload="", error="", song_link="", linked_song="")
        return render_template("no_access.html")

    elif 'uploadSongForm' in request.files:
        uploaded_song = request.files['uploadSongForm']
        filename = uploaded_song.filename
        filename = "".join(i for i in filename if i not in "/\\:*?<>|.'")
        filename = filename[:-3] + '.' + filename[-3:]

        if filename != '':
            ########## Save file on server ##########
            dot_index = filename.rfind('.')
            raw_filename = filename[0:dot_index]
            file_format = filename[dot_index:len(filename)+1] #.mp3

            if os.path.isfile('static/uploads/audio/' + filename) :
                i = 1

                while os.path.isfile('static/uploads/audio/' + raw_filename + '(' + str(i) + ')' + file_format):
                    i += 1

                raw_filename += '(' + str(i) + ')'
                filename = raw_filename + file_format

            uploaded_song.save(os.path.join('static/uploads/audio', filename))

            ########## Convert to .mp3 if needed ##########
            if file_format != '.mp3':
                conversions.convert_to_mp3(filename)
                os.remove(os.path.join('static/uploads/audio', filename))
                filename = raw_filename + '.mp3'
            else:
                set_delete_timer(os.path.join('static/uploads/audio', filename))

            encoded = encode(filename)
            duration = MP3('static/uploads/audio/' + filename).info.length

            if duration > 300:
                os.remove(os.path.join('static/uploads/audio', filename))
                error = "Audio is longer than 5 minutes"
            elif duration < 5:
                os.remove(os.path.join('static/uploads/audio', filename))
                error = "Audio is shorter than 5 seconds"
            else:
                error = ""
            return render_template("audio.html", user_upload=filename, encoded=encoded, error=error, song_link="", linked_song="")
        else:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("audio.html", user_upload="", encoded="", error="", song_link="", linked_song="")

    elif 'linkSongForm' in request.form:
        link = request.form['linkSongForm']

        try:
            youtube = YouTube(link)
            if YouTube(link).length > 300:
                session['files'] = {'matching': session['files']['matching']}
                return render_template("audio.html", user_upload="", encoded="", error="Audio is longer than 5 minutes", song_link=link, linked_song="")
            elif YouTube(link).length < 5:
                session['files'] = {'matching': session['files']['matching']}
                return render_template("audio.html", user_upload="", encoded="", error="Audio is shorter than 5 seconds", song_link=link, linked_song="")
            else:
                ########## Save file on server ##########
                title = YouTube(link).streams.first().title
                title = "".join(i for i in title if i not in "/\\:*?<>|.'")
                video_title = title

                if os.path.isfile('static/uploads/video/' + video_title + '.mp4'):
                    i = 1

                    while os.path.isfile('static/uploads/video/' + video_title + '(' + str(i) + ')' + '.mp4'):
                        i += 1

                    video_title += '(' + str(i) + ')'

                resolutions = ["1080p", "720p", "360p", "240p", "144p"]
                res = ""
                videos = []

                for resolution in resolutions:
                    videos = YouTube(link).streams.filter(res=resolution, progressive=True)

                    if len(videos) > 0:
                        res = resolution
                        break

                video = videos[0].download(output_path='static/uploads/video/', filename=video_title)

                ########## Convert YouTube video to .mp3 ##########
                audio_title = title

                if os.path.isfile('static/uploads/audio/' + audio_title + '.mp3'):
                    i = 1

                    while os.path.isfile('static/uploads/audio/' + audio_title + '(' + str(i) + ')' + '.mp3'):
                        i += 1

                    audio_title += '(' + str(i) + ')'

                raw_audio = ffmpeg.input('static/uploads/video/' + video_title + '.mp4')
                processed_video = ffmpeg.output(raw_audio, 'static/uploads/audio/' + audio_title + '.mp3')
                encoded = encode(audio_title+'.mp3')
                ffmpeg.run(processed_video)
                os.remove('static/uploads/video/' + video_title + '.mp4')
                set_delete_timer('static/uploads/audio/' + audio_title + '.mp3')
                return render_template("audio.html", user_upload="", encoded=encoded, error="", song_link="", linked_song=audio_title+'.mp3')

        except pytube.exceptions.RegexMatchError:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("audio.html", user_upload="", encoded="", error="Invalid link", song_link=link, linked_song="")

        except pytube.exceptions.ExtractError:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("audio.html", user_upload="", encoded="", error="An extraction Error occurred", song_link=link, linked_song="")

        except pytube.exceptions.VideoUnavailable:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("audio.html", user_upload="", encoded="", error="The video is unavailabe", song_link=link, linked_song="")
    else:
        session['files'] = {'matching': session['files']['matching']}
        return redirect(url_for('audio'))


@app.route("/video", methods=['GET', 'POST'])
def video():
    '''route to upload a video file'''
    safari = str(request.user_agent.browser == "safari")
    reset_camera()
    if request.method == 'GET':
        if 'files' in session and 'matching' in session['files']:
            return render_template("video.html", matching="", user_upload="", error="", video_link="", linked_video="", safari=safari)
        return render_template("no_access.html")
    elif 'matching' in request.form:
        matching = request.form.get('matching')
        return render_template("video.html", matching=matching, user_upload="", error="", video_link="", linked_video="", safari=safari)
    elif 'userUpload' in request.files:
        uploaded_video = request.files['userUpload']
        filename = uploaded_video.filename
        filename = "".join(i for i in filename if i not in "/\\:*?<>|.'")
        filename = filename[:-3] + '.' + filename[-3:]

        if filename != '':
            ########## Save file on server ##########
            dot_index = filename.rfind('.')
            raw_filename = filename[0:dot_index]
            file_format = filename[dot_index:len(filename)+1] #.mp4

            if os.path.isfile('static/uploads/video/' + filename) :
                i = 1

                while os.path.isfile('static/uploads/video/' + raw_filename + '(' + str(i) + ')' + file_format):
                    i += 1

                raw_filename += '(' + str(i) + ')'
                filename = raw_filename + file_format

            uploaded_video.save(os.path.join('static/uploads/video/', filename))

            duration = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", "static/uploads/video/" + filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            if float(duration.stdout) > 300:
                os.remove(os.path.join('static/uploads/video/', filename))
                error = "Video is longer than 5 minutes"
            elif float(duration.stdout) < 5:
                os.remove(os.path.join('static/uploads/video/', filename))
                error = "Video is shorter than 5 seconds"
            else:
                ########## Convert to .mp4 if needed ##########
                if file_format != '.mp4':
                    conversions.convert_to_mp4(filename)
                    os.remove(os.path.join('static/uploads/video/', filename))
                    filename = raw_filename + '.mp4'
                else:
                    set_delete_timer(os.path.join('static/uploads/video/', filename))

                error = ""

            encoded = encode(filename)

            return render_template("video.html", user_upload=filename, encoded=encoded, error=error, video_link="", linked_video="", safari=safari)
        else:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("video.html", user_upload="", encoded="", error="", video_link="", linked_video="", safari=safari)
    elif 'videoLink' in request.form:
        link = request.form['videoLink']

        try:
            youtube = YouTube(link)

            if YouTube(link).length > 300:
                session['files'] = {'matching': session['files']['matching']}
                return render_template("video.html", user_upload="", encoded="", error="Video is longer than 5 minutes", video_link=link, linked_video="", safari=safari)
            elif YouTube(link).length < 5:
                session['files'] = {'matching': session['files']['matching']}
                return render_template("video.html", user_upload="", encoded="", error="Video is shorter than 5 seconds", video_link=link, linked_video="", safari=safari)
            else:
                ########## Save file on server ##########
                title = YouTube(link).streams.first().title
                title = "".join(i for i in title if i not in "/\\:*?<>|.'")

                if os.path.isfile('static/uploads/video/' + title + '.mp4'):
                    i = 1

                    while os.path.isfile('static/uploads/video/' + title + '(' + str(i) + ')' + '.mp4'):
                        i += 1
                    title += '(' + str(i) + ')'

                resolutions = ["1080p", "720p", "360p", "240p", "144p"]
                res = ""
                videos = []

                for resolution in resolutions:
                    videos = YouTube(link).streams.filter(res=resolution, progressive=True)

                    if len(videos) > 0:
                        res = resolution
                        break

                video = videos[0].download(output_path='static/uploads/video/', filename=title)
                set_delete_timer('static/uploads/video/' + title + '.mp4')
                encoded = encode(title+'.mp4')

                return render_template("video.html", user_upload="", encoded=encoded, error="", video_link="", linked_video=title+'.mp4', safari=safari)

        except pytube.exceptions.RegexMatchError:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("video.html", user_upload="", encoded="", error="Invalid link", video_link=link, linked_video="", safari=safari)

        except pytube.exceptions.ExtractError:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("video.html", user_upload="", encoded="", error="An extraction Error occurred", video_link=link, linked_video="", safari=safari)

        except pytube.exceptions.VideoUnavailable:
            session['files'] = {'matching': session['files']['matching']}
            return render_template("video.html", user_upload="", encoded="", error="The video is unavailabe", video_link=link, linked_video="", safari=safari)
    else:
        session['files'] = {'matching': session['files']['matching']}
        return redirect(url_for('video'))


@app.route("/camera", methods=['GET', 'POST'])
def camera():
    '''route for recording a video'''
    if request.user_agent.browser == 'safari':
        return render_template('page_not_found.html')
    elif 'files' not in session or 'matching' not in session['files']:
        return render_template('no_access.html')

    if request.method == 'POST':
        session['files'] = {'matching': session['files']['matching']}
        reset_camera()

    return render_template("camera.html")


@app.route('/_record_status', methods=['GET', 'POST'])
def record_status():
    '''setting the status of video recording'''
    global global_frame
    global video_camera

    if request.method == 'GET':
        return render_template("page_not_found.html")

    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")


def video_stream():
    '''initiating video streaming for recording a video'''
    global video_camera
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()

    while video_camera != None:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


@app.route('/_video_viewer')
def video_viewer():
    '''returning the "visual" video stream'''
    if request.user_agent.browser == 'safari' or 'files' not in session or 'matching' not in session['files']:
        return render_template('page_not_found.html')
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/confirm_recording')
def confirm_recording():
    '''rout to watch and confirm webcam recording'''
    reset_camera()
    if request.user_agent.browser == 'safari':
        return render_template('page_not_found.html')

    if not 'files' in session or not 'recording_name' in session['files'] or not 'matching' in session['files']:
        return render_template('no_access.html')

    recording = decode(session['files']['recording_name'])
    session['files'] = {'matching': session['files']['matching']}
    encoded = encode(recording)

    return render_template("confirm_recording.html", recording=recording, encoded=encoded)


@app.route("/waiting/<data_type>/<name>")
def waiting(data_type, name):
    '''waiting route returning a loading screen to the user'''
    filename = decode(name)

    if not os.path.isfile('static/uploads/' + data_type + '/' + filename):
        return render_template("page_not_found.html")

    elif not 'files' in session or not 'matching' in session['files']:
        return render_template("no_access.html")

    return render_template("waiting.html", type=data_type, name=filename, result="")


@app.route("/_extract_features")
def extract_features():
    '''extracting features from user input

    extract frames (for video) or 5 second segments (for audio) from the user input
    extract feature vectors from frames/segments
    get values for valence and arousal from feature vectors
    write these values to a .txt file (inside model)
    '''
    if None in [request.args.get('data_type'), request.args.get('name')]:
        return render_template("page_not_found.html")

    ########## Get features ##########
    data_type = request.args.get('data_type')
    name = request.args.get('name')

    if data_type == "video":
        folder, extension = conversions.get_frames('static/uploads/video/' + name)

        if folder == "Error":
            jsonify(status="error")
        else:
            if extension < 64 and extension >= 2:
                os.system("python3.7 Model/ResNet50/feature_extraction_ResNet50.py '" + folder + "' '64'")
                os.system("python3.7 Model/I3D_RGB/feature_extraction_I3D_RGB.py '" + folder + "' '64'")
                os.system("python3.7 Model/FlowNetS/feature_extraction_FlowNetS.py '" + folder + "' '64'")
                os.system("python3.7 Model/AttendAffectNet/video_model_arousal.py '" + folder + "' '64'")
                os.system("python3.7 Model/AttendAffectNet/video_model_valence.py '" + folder + "' '64'")
                os.system("python3.7 Model/ResNet50/feature_extraction_ResNet50.py '" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/I3D_RGB/feature_extraction_I3D_RGB.py '" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/FlowNetS/feature_extraction_FlowNetS.py '" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/AttendAffectNet/video_model_arousal.py '" + folder + "' '" + str(extension) + "'")
                os.system("python3.7 Model/AttendAffectNet/video_model_valence.py '" + folder + "' '" + str(extension) + "'")
                return video_matching(name, folder)
            else:
                os.system("python3.7 Model/ResNet50/feature_extraction_ResNet50.py '" + folder + "' '64'")
                os.system("python3.7 Model/I3D_RGB/feature_extraction_I3D_RGB.py '" + folder + "' '64'")
                os.system("python3.7 Model/FlowNetS/feature_extraction_FlowNetS.py '" + folder + "' '64'")
                os.system("python3.7 Model/AttendAffectNet/video_model_arousal.py '" + folder + "' '64'")
                os.system("python3.7 Model/AttendAffectNet/video_model_valence.py '" + folder + "' '64'")
                return video_matching(name, folder)
    elif data_type == "audio":
        folder = conversions.get_audio_clips('static/uploads/audio/' + name)

        if folder == "Error":
            jsonify(status="error")
        else:
            os.system("python3.7 Model/VGGish/feature_extraction_VGGish.py '" + folder + "'")
            os.system("python3.7 Model/OpenSMILE/feature_extraction_emobase2010_320_40.py '" + folder + "'")
            os.system("python3.7 Model/AttendAffectNet/audio_model_arousal.py '" + folder + "'")
            os.system("python3.7 Model/AttendAffectNet/audio_model_valence.py '" + folder + "'")
            return audio_matching(name, folder)
    else:
        return jsonify(status="error")


def audio_matching(audio, folder):
    '''find matching video files for audio and define the video resolution'''
    ### Get Values ###
    values = 'static/audioClips' + folder+ '/values.txt'

    with open(values) as f:
        duration = f.readline().strip()
        arousal_string = f.readline().strip()
        valence_string = f.readline().strip()

    arousal = [float(value.strip()) for value in arousal_string.split(', ')]
    valence = [float(value.strip()) for value in valence_string.split(', ')]

    ########## Matching ##########
    matching_method = session['files']['matching']

    if "approx" in matching_method:
        mode = "standard" if matching_method == "approx" else "mse"

        if matching_method == "approx_w":
            mode = "weighted"

        matching_files, matching_indexes = approximation_matching(folder, "audio", mode)

    elif "offline" in matching_method:
        mode = "standard" if matching_method == "offline" else "mse"

        if matching_method == "offline_w":
            mode = "weighted"

        matching_files, matching_indexes = offline_matching (folder, "audio", mode)

    else:
        matching_files, matching_indexes = random_matching(folder, "audio")

    for i in range(0, len(matching_files)):
        matching_files[i] = encode(matching_files[i])

    session['files'] = {'matching_files': matching_files, 'matching_indexes': matching_indexes}

    ########## Get video resolution ##########
    max_width = 0.0
    max_height = 0.0

    for video in session['files']['matching_files']:
        props = get_video_properties('static/datasets/video/' + decode(video) + '.mp4')
        if props['width'] > max_width:
            max_width = props['width']
        if props['height'] > max_height:
            max_height = props['height']

    resolution = {"width": max_width, "height": max_height}

    return merge_audio_result(audio, resolution)


def merge_audio_result(audio, resolution):
    '''merging files to final result

    merging video files of matching result to one video
    merging this video and the user input to final result
    '''
    if not 'files' in session or not 'matching_files' in session['files']:
        return render_template('no_access.html')

    ########## Merge matching video files ##########
    input_video_file_name = ""

    if len(session['files']['matching_files']) > 1:
        command = ["ffmpeg"]

        for video in session['files']['matching_files']:
            command += ["-i", "static/datasets/video/" + decode(video) + ".mp4"]

        command.append("-filter_complex")
        scaling = ""

        ########## Construct command for merging video files ##########
        for i in range(0, len(session['files']['matching_files'])):
            start_time = str(int(session['files']['matching_indexes'][i][0] * 5))
            end_time = str(int(session['files']['matching_indexes'][i][1] * 5))
            scale = str(resolution["width"]) + ":" + str(resolution["height"])
            scaling += "[" + str(i) + "]trim=" + start_time + ":" + end_time + ",setpts=PTS-STARTPTS,scale=" + scale + ":force_original_aspect_ratio=decrease,pad=" + scale + ":(ow-iw)/2:(oh-ih)/2,setsar=1[" + str(i) + "v];"

            if i < len(session['files']['matching_files']) - 1:
                scaling += " "

        for i in range(0, len(session['files']['matching_files'])):
            scaling += "[" + str(i) + "v] "

        scaling += "concat=n=" + str(len(session['files']['matching_files'])) + ":v=1 [v]"
        i = 1
        input_video_file_name = "matching_video(" + str(i) + ")"

        while os.path.isfile('static/uploads/video/' + input_video_file_name + '.mp4'):
            i += 1
            input_video_file_name = "matching_video(" + str(i) + ")"

        command += [scaling, "-map", "[v]", "-vsync", "2", "static/uploads/video/" + input_video_file_name + ".mp4"]
        subprocess.call(command)
        set_delete_timer("static/uploads/video/" + input_video_file_name + ".mp4")
        folder = "/uploads"

    else:
        folder = "/datasets"
        input_video_file_name = decode(session['files']['matching_files'][0])

    ########## Merge final result ##########
    input_audio = ffmpeg.input('static/uploads/audio/' + audio).audio
    input_video = ffmpeg.input('static' + folder + '/video/' + input_video_file_name + '.mp4').video
    i = 1
    filename = "matching_result(" + str(i) + ")"

    while os.path.isfile('static/results/' + filename + '.mp4'):
        i += 1
        filename = "matching_result(" + str(i) + ")"

    session['files'] = {'matching_files': session['files']['matching_files'], 'result': encode(filename)}
    ffmpeg.output(input_audio, input_video, 'static/results/' + filename + '.mp4', shortest=None).run()
    set_delete_timer('static/results/' + filename + '.mp4')

    return jsonify(status="audio")


def video_matching(video, folder):
    '''Find matching audi files for video'''
    ########## Get values ##########
    values = 'static/frames' + folder + '/values.txt'

    with open(values) as f:
        duration = f.readline().strip()
        arousal_string = f.readline().strip()
        valence_string = f.readline().strip()

    arousal = [float(value.strip()) for value in arousal_string.split(', ')]
    valence = [float(value.strip()) for value in valence_string.split(', ')]

    ########## Matching ##########
    matching_method = session['files']['matching']

    if "approx" in matching_method:
        mode = "standard" if matching_method == "approx" else "mse"

        if matching_method == "approx_w":
            mode = "weighted"

        matching_files, matching_indexes = approximation_matching(folder, "video", mode)

    elif "offline" in matching_method:
        mode = "standard" if matching_method == "offline" else "mse"

        if matching_method == "offline_w":
            mode = "weighted"

        matching_files, matching_indexes = offline_matching (folder, "video", mode)

    else:
        matching_files, matching_indexes = random_matching(folder, "video")

    for i in range(0, len(matching_files)):
        matching_files[i] = encode(matching_files[i])

    session['files'] = {'matching_files': matching_files, 'matching_indexes': matching_indexes}

    return merge_video_result(video)


def merge_video_result(video):
    '''merging files to final result

    merging audio files of matching result to one audio file
    merging this audio and the user input to final result
    '''
    if not 'files' in session or not 'matching_files' in session['files'] or not 'matching_indexes' in session['files']:
        return render_template('no_access.html')

    ########## Merge matching audio files ##########
    matching_audio_segments = []

    for i in range(0, len(session['files']['matching_files'])):
        j = 1
        file = 'static/matching_audio/matching_audio(' + str(j) + ').mp3'

        while os.path.isfile(file):
            j += 1
            file = 'static/matching_audio/matching_audio(' + str(j) + ').mp3'

        start_time = str(session['files']['matching_indexes'][i][0] * 5) + '.0'
        duration = (session['files']['matching_indexes'][i][1] - session['files']['matching_indexes'][i][0] + 1) * 5
        end_time = str(duration) + '.0'
        subprocess.call(['ffmpeg', '-ss', start_time, '-i', 'static/datasets/audio/' + decode(session['files']['matching_files'][i]) + '.mp3', '-c', 'copy', '-t', end_time, file])
        set_delete_timer(file)
        matching_audio_segments.append(file)

    matching_audio = AudioSegment.from_file(matching_audio_segments[0])
    input_audio_file = matching_audio_segments[0]

    if len(session['files']['matching_files']) > 1:

        for i in range(1, len(session['files']['matching_files'])):
            matching_audio += AudioSegment.from_file(matching_audio_segments[i])

        i = 1
        input_audio_file = 'static/uploads/audio/matching_audio(' + str(i) + ').mp3'

        while os.path.isfile(input_audio_file):
            i += 1
            input_audio_file = 'static/uploads/audio/matching_audio(' + str(i) + ').mp3'

        matching_audio.export(input_audio_file, format='mp3')
        set_delete_timer(input_audio_file)

    ########## Merge final result ##########
    input_audio = ffmpeg.input(input_audio_file).audio
    input_video = ffmpeg.input('static/uploads/video/' + video).video
    i = 1
    filename = "matching_result(" + str(i) + ")"

    while os.path.isfile('static/results/' + filename + '.mp4'):
        i += 1
        filename = "matching_result(" + str(i) + ")"

    session['files'] = {'matching_files': session['files']['matching_files'], 'result': encode(filename)}
    ffmpeg.output(input_audio, input_video, 'static/results/' + filename + '.mp4', shortest=None).run()
    set_delete_timer('static/results/' + filename + '.mp4')

    return jsonify(status="video")


@app.route("/audio_result")
def audio_result():
    '''route for presenting result of matching for audio input'''
    if 'files' not in session or 'result' not in session['files'] or 'matching_files' not in session['files']:
        return render_template("no_access.html")

    ########## Get files found in matching ##########
    filenames = []

    if len(session['files']['matching_files']) == 1:

        with open('static/datasets/video/metadata.csv', newline='') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                if row[0] == decode(session['files']['matching_files'][0]) + '.mp4':
                    filenames = [{"name": decode(session['files']['matching_files'][0]), "number": "", "title": row[1], "source": row[2]}]
    else:
        numbers = ["first ", "second ", "third ", "fourth ", "fifth ", "sixth ", "seventh ", "eighth ", "ninth ", "tenth "]
        matching_files_decoded = []

        for i in range(0, len(session['files']['matching_files'])):
            matching_files.append(decode(session['files']['matching_files'][i]) + '.mp4')

        video_properties = {}

        with open('static/datasets/video/metadata.csv', newline='') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                if row[0] in matching_files_decoded:
                    video_properties[row[0]] == {'title': row[1], 'source': row[2]}

        for i in range(0, len(session['files']['matching_files'])):
            name = decode(session['files']['matching_files'][i])
            filenames.append({"name": name, "number": numbers[i], "title": video_properties[name]['title'], "source": video_properties[name]['source']})

    result_saved = decode(session['files']['result'])

    return render_template("audio_result.html", video_names=filenames, result=result_saved+'.mp4')


@app.route("/video_result")
def video_result():
    '''route for presenting result of matching for video input'''
    if 'files' not in session or 'result' not in session['files'] or 'matching_files' not in session['files']:
        return render_template("no_access.html")

    ########## Get files found in matching ##########
    filenames = []

    if len(session['files']['matching_files']) == 1:

        with open('static/datasets/audio/metadata.csv', newline='') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                if row[0] == decode(session['files']['matching_files'][0]) + '.mp3':
                    filenames = [{"name": decode(session['files']['matching_files'][0]), "number": "", "title": row[1], "artist": row[2]}]
    else:
        numbers = ["first ", "second ", "third ", "fourth ", "fifth ", "sixth ", "seventh ", "eighth ", "ninth ", "tenth "]
        matching_files_decoded = []

        for i in range(0, len(session['files']['matching_files'])):
            matching_files.append(decode(session['files']['matching_files'][i]) + '.mp3')

        song_properties = {}

        with open('static/datasets/audio/metadata.csv', newline='') as csv_file:
            reader = csv.reader(csv_file)

            for row in reader:
                if row[0] in matching_files_decoded:
                    song_properties[row[0]] = {'title': row[1], 'artist': row[2]}

        for i in range(0, len(session['files']['matching_files'])):
            name = decode(session['files']['matching_files'][i])
            filenames.append({"name": name, "number": numbers[i], "title": song_properties[name]['title'], "artist": song_properties[name]['artist']})

    result_saved = decode(session['files']['result'])

    return render_template("video_result.html", files=filenames, result=result_saved+'.mp4')


@app.route("/download_result/<path:filename>", methods=['GET', 'POST'])
def download_result(filename):
    '''route for downloading the final result'''
    path = decode(path)
    uploads = os.path.join('static/results')

    return send_from_directory(uploads, path, as_attachment=True)


@app.route("/download_audio/<path:filename>", methods=['GET', 'POST'])
def download_audio(filename):
    '''route for downloading the matched audio file'''
    path = decode(path)
    uploads = os.path.join('static/datasets/audio')

    return send_from_directory(uploads, path, as_attachment=True)


@app.route("/download_video/<path:filename>", methods=['GET', 'POST'])
def download_video(filename):
    '''route for downloading the matched video file'''
    path = decode(path)
    uploads = os.path.join('static/datasets/video')

    return send_from_directory(uploads, path, as_attachment=True)


@app.errorhandler(404)
def page_not_found(e):
    '''custom html page for 404 error'''
    return render_template('page_not_found.html'), 404


@app.errorhandler(500)
def server_error(e):
    '''500 error'''
    reset_camera()


def reset_camera():
    '''resetting the camera'''
    global video_camera

    if video_camera != None:
        if video_camera.name != None:
            session['files'] = {'recording_name': encode(video_camera.name), 'matching': session['files']['matching']}
        video_camera = None


@app.route('/favicon.ico')
def favicon():
    '''returning the favicon of the website'''
    return send_from_directory('static/', 'favicon.ico')


if __name__ == "__main__":
    app.run(debug = True)
