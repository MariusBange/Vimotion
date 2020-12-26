/**
script for audio.html
*/
$(function() {
    /**
    after loading the page assign functions to buttons and setting properties
    */
    $('#uploadSongForm').on('change', (function() {
        /**
        enables the upload button for user uploads when user selected a file
        */
        $('#uploadSongButton').prop('disabled', false);
    }));
    $('#linkSongForm').on('input', (function() {
        /**
        enables the upload button for linked audio when user entered something
        and disabled the upload button, if input is empty
        */
        if ($('#linkSongForm').val().length < 1) {
            $("#linkSongButton").prop('disabled', true);
        }
        else{
            $("#linkSongButton").prop('disabled', false);
        }
    }));
    if ($("#linkSongForm").val() == "") {
        $("#linkSongButton").prop('disabled', true);
    }
    else {
        $("#linkSongButton").prop('disabled', false);
    }
});
