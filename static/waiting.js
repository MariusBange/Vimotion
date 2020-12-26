/**
script for waiting.html
*/
$(function() {
    /**
    after loading the page instantly send get request to merge the final result and redirect to the corresponding url
    */
    $.getJSON('/_extract_features', {
        data_type: document.getElementById('type').textContent,
        name: document.getElementById('name').textContent,
    }, function(data) {
        var status = data.status;
        if (status == "audio") {
            window.location='/audio_result';
        }
        else if (status == "video") {
            window.location='/video_result';
        }
        else {
            alert('There was an error. In case of video: Please make sure the video has at least 12.8 frames per second.');
        }
    });
});
