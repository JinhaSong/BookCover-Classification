import os
import yaml

from utils import Logging


def check_file(path):
    assert os.path.exists(path), Logging.e("File Not Found: \"{}\"".format(path))


def check_is_empty_dir(path):
    assert not len(os.listdir(path)), Logging.e("Target Directory Not Empty: \"{}\"".format(path))


def check_dir(path, with_create=True):
    is_exist = os.path.exists(path) and os.path.isdir(path)
    if with_create:
        if not is_exist:
            os.makedirs(path)
    else:
        assert is_exist, Logging.e("Directory Not Found: \"{}\"".format(path))


def read_yaml(path):
    check_file(path)
    with open(path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        return data


def write_data(path, ldata):
    try:
        with open(path, "w") as file:
            for line in ldata:
                file.write(line)
            file.close()
    except:
        print(Logging.e("Cannot write file: \"{}\"".format(path)))
