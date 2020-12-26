/**
script for video.html
*/
$(function() {
    /**
    after loading the page assign functions to buttons and setting properties
    */
    $('#userUpload').on('change', (function() {
        /**
        enables the upload button for user uploads when user selected a file
        */
        $('#upload_user_file').prop('disabled', false);
    }));
    $('#videoLink').on('input', (function() {
        /**
        enables the upload button for linked videos when user entered something
        and disabled the upload button, if input is empty
        */
        if ($('#videoLink').val().length < 1) {
            $("#upload_yt_btn").prop('disabled', true);
        }
        else{
            $("#upload_yt_btn").prop('disabled', false);
        }
    }));
    if ($("#videoLink").val() == "") {
        $("#upload_yt_btn").prop('disabled', true);
    }
    else {
        $("#upload_yt_btn").prop('disabled', false);
    }
});
