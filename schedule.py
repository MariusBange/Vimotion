'''deletion of file after 24 hours'''
import os
import threading
import shutil

files = []

def delete():
    '''deletes the first file in files from server'''
    file = files[0]
    del files[0]
    if os.path.isdir(file):
        shutil.rmtree(file, ignore_errors=True)
    elif os.path.isfile(file):
        os.remove(file)


def set_delete_timer(file):
    '''starts timer to delete file after 24 hours'''
    global files
    files.append(file)
    timer = threading.Timer(864000, delete)
    timer.start()
