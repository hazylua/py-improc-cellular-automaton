from os import listdir, path, unlink, mkdir
from shutil import rmtree

def clear_dir(dir_path):
    for filename in listdir(dir_path):
        file_path = path.join(dir_path, filename)
        try:
            if path.isfile(file_path) or path.islink(file_path):
                unlink(file_path)
            elif path.isdir(file_path):
                rmtree(file_path)
            print("Done clearing.")
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason {e}')

def check_dir(dir_path):
    if path.exists(dir_path):
        print('This path exists.')
    else:
        print('This path does not exist. Creating it now.')
        mkdir(dir_path)