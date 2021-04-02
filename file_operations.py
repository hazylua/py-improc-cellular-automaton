from os import listdir, path, unlink, mkdir
from shutil import rmtree

def clear_dir(dir_path):
    print('Clearing path.')
    if check_dir(dir_path, make_dir=False):
        if(len(listdir(dir_path)) == 0):
            print('Already cleared.')
        else:
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

def check_dir(dir_path, make_dir):
    print('Checking if path exists...')
    if path.exists(dir_path):
        print('This path exists.')
        return True
    elif make_dir == True:
        try:
            print('This path does not exist. Creating it now.')
            mkdir(dir_path)
            return True
        except Exception as e:
            print(f'Failed to create {dir_path}.\nReason: {e}')
            return False
    else:
        print('This path does not exist.')
        return False