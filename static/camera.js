/**
script for camera.html
*/
var buttonCancel = document.getElementById("cancel");
var buttonRecord = document.getElementById("record");
var buttonContinue = document.getElementById("continue");
window.onpagehide = confirmExit;
var recordingTimer;
var timer;
var time;

buttonCancel.onclick = function() {
    /**
    on cancle button click if currently recording stop recording
    */
    if ((buttonRecord.textContent == "Stop Recording") || (buttonRecord.disabled == true)) {
        stopRecording();
        confirmExit();
    }
};

buttonRecord.onclick = function() {
    /**
    toggling record button text
    */
    if (buttonRecord.textContent == "Record") {
        startRecording();
    }
    else if (buttonRecord.textContent == "Stop Recording") {
        stopRecording();
    }
};

function startRecording() {
    /**
    updating interface and start recording camera
    */
    buttonContinue.disabled = true;
    buttonRecord.disabled = true;
    buttonRecord.textContent = "Stop Recording";
    document.getElementById("timer_up").innerHTML = "00:00";
    document.getElementById("timer_down").innerHTML = "-05:00";
    document.getElementById("recording_time").innerHTML = "";
    setTimer("start");

    recordingTimer = setTimeout(stopRecording, 300000);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/_record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "true" }));
};

function stopRecording() {
    /**
    updating interface and stop recording camera
    */
    buttonContinue.disabled = false;
    buttonRecord.textContent = "Record";
    setTimer("stop");

    if (recordingTimer != null) {
        recordingTimer = null;
    };

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/_record_status");
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    xhr.send(JSON.stringify({ status: "false" }));
};

function setTimer(action) {
    /**
    handles the timer for video recording
    */
    if (action == "start") {
        var startTime = new Date().getTime();
        var currentDate = new Date();
        currentDate.getSeconds( currentDate.getSeconds() + 300);
        var stopTime = currentDate.getTime();
        time = 0;

        timer = setInterval(function() {

            var currentTime = new Date().getTime();
            ++time;
            if (time == 5) {
                buttonRecord.disabled = false;
            }

            var countdown = 300 - time;

            var minutes_up = Math.floor(time/60);
            var seconds_up = time % 60;
            var minutes_down = Math.floor(countdown/60);
            var seconds_down = countdown % 60;

            if (seconds_up > 9) {
                document.getElementById("timer_up").innerHTML = "0" + minutes_up + ":" + seconds_up;
            }
            else {
                document.getElementById("timer_up").innerHTML = "0" + minutes_up + ":0" + seconds_up;
            }
            if (seconds_down > 9) {
                document.getElementById("timer_down").innerHTML = "-0" + minutes_down + ":" + seconds_down;
            }
            else {
                document.getElementById("timer_down").innerHTML = "-0" + minutes_down + ":0" + seconds_down;
            }

            if (time >= 300) {
                clearInterval(timer);
                document.getElementById("timer_up").innerHTML = "";
                document.getElementById("timer_down").innerHTML = "";
                document.getElementById("recording_time").innerHTML = "05:00";
            };
        }, 1000);
    }
    else if (action == "stop") {
        clearInterval(timer);
        document.getElementById("timer_up").innerHTML = "";
        document.getElementById("timer_down").innerHTML = "";

        var minutes = Math.floor(time/60);
        var seconds = time % 60;

        if (seconds > 9) {
            document.getElementById("recording_time").innerHTML = "0" + minutes + ":" + seconds;
        }
        else {
            document.getElementById("recording_time").innerHTML = "0" + minutes + ":0" + seconds;
        }
    };
}

function confirmExit() {
    /**
    sends post request to stop camera when leaving site
    doesn't work on safari after first call because of back-forward cache
    */
    if (buttonRecord.textContent == "Stop Recording") {
        stopRecording();
    }
    $.ajax({
        type : "POST",
        url : '/camera'
    });
    return undefined;
};
