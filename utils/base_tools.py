import os


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file