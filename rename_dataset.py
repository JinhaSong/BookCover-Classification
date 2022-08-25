import json
import argparse
from utils.common import *
from utils.file_utils import *
from dataset.rename import RenameUtil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='rename_dataset.py')
    parser.add_argument('--origin-dir', type=str, help='original dataset directory path')
    parser.add_argument('--target-dir', type=str, help='target directory path to split into train, test, val')
    parser.add_argument('--min-cls-img-nb', type=int, default=1000, help='minimum number of images that a class must contain')
    parser.add_argument('--dataset-ratio', type=str, default="1,1,4", help='ratio to use when splitting by test, val, train')
    parser.add_argument('--dataset-class-info', type=str, default="data/bookcover_class.yaml", help='dataset class information file path')
    parser.add_argument('--is-filter', action="store_true", help='is filtering class by number of images')
    parser.add_argument('--save-dataset-info', action="store_true", help='save dataset information')
    parser.add_argument('--debug', action="store_true", help='debug')

    opt = parser.parse_args()
    origin_dir = opt.origin_dir
    target_dir = opt.target_dir
    min_cls_img_nb = opt.min_cls_img_nb
    dataset_ratio = list(map(int, opt.dataset_ratio.split(",")))
    is_filter = opt.is_filter
    save_dataset_info = opt.save_dataset_info
    dataset_class_info = read_yaml(opt.dataset_class_info)
    debug = opt.debug

    print(Logging.i("Argument Info:"))
    print(Logging.s(f"\torigin dir: {origin_dir}"))
    print(Logging.s(f"\ttarget dir: {target_dir}"))
    if is_filter:
        print(Logging.s(f"\tminimum number of image that class must contain: {min_cls_img_nb}"))
    print(Logging.s(f"\tdataset ratio: {dataset_ratio}"))

    rename_util = RenameUtil(dataset_class_info, origin_dir, target_dir, dataset_ratio, is_filter, min_cls_img_nb, debug)
    rename_util.rename_dataset()
    rename_util.save_data()
    if save_dataset_info:
        rename_util.save_dataset_info()
