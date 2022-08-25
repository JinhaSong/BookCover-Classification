import os
import shutil
from utils.file_utils import *


class RenameUtil:
    def __init__(self, dataset_class_info, origin_dir, target_dir, dataset_ratio, is_filter=False, min_cls_img_nb=1000, debug=False):
        self.eng_class = dataset_class_info["eng_class"]
        self.kor_class = dataset_class_info["kor_class"]
        self.nc = dataset_class_info["nc"]
        self.origin_dir = origin_dir
        self.target_dir = target_dir
        self.is_filter = is_filter
        self.min_cls_img_nb = min_cls_img_nb
        self.filter_kor_class = []
        self.filter_eng_class = []
        self.image_list = []
        self.dataset_type = ["test", "val", "train"]
        self.dataset_ratio = dataset_ratio
        self.cls_nb_img = []
        self.debug = debug
        self.rename_dataset_path = ""
        self.data_path = ""

    def rename_dataset(self):
        print(Logging.i("Processing...(Rename)"))
        ret = self.create_dataset_dir()
        assert ret, Logging.e("Creating dataset directories is failed")

        ret = self.copy_images2target_dir()
        assert ret, Logging.e("Creating and renaming dataset is failed")
        print(Logging.i("Renaming is done."))

    def create_dataset_dir(self):
        try:
            check_dir(self.origin_dir, with_create=False)
            check_dir(self.target_dir, with_create=True)
            check_is_empty_dir(self.target_dir)

            for dtype in self.dataset_type:
                check_dir(os.path.join(self.target_dir, dtype), with_create=True)

            for i, kcls in enumerate(self.kor_class):
                for d, dtype in enumerate(self.dataset_type):
                    kor_class_path = os.path.join(self.origin_dir, kcls)
                    eng_class_path = os.path.join(self.target_dir, dtype, self.eng_class[i])
                    nb_image = len(os.listdir(kor_class_path))
                    if self.is_filter:
                        if nb_image >= self.min_cls_img_nb and dtype == "train":
                            self.filter_kor_class.append(self.kor_class[i])
                            self.filter_eng_class.append(self.eng_class[i])
                        os.makedirs(eng_class_path)
                    else:
                        if dtype == "train" :
                            self.filter_kor_class.append(self.kor_class[i])
                            self.filter_eng_class.append(self.eng_class[i])
                        os.makedirs(eng_class_path)
            return True
        except:
            return False

    def copy_images2target_dir(self):
        try:
            total_fold = sum(self.dataset_ratio)
            for i, kcls in enumerate(self.filter_kor_class):
                cls_dir = os.path.join(self.origin_dir, kcls)
                cls_img_list = os.listdir(cls_dir)
                cls_fold_img_nb = int(len(cls_img_list) / total_fold)
                cls_img_counts = [
                    [0, cls_fold_img_nb * self.dataset_ratio[0]], # test
                    [cls_fold_img_nb * self.dataset_ratio[0], cls_fold_img_nb * (self.dataset_ratio[0] + self.dataset_ratio[1])], # val
                    [cls_fold_img_nb * (self.dataset_ratio[0] + self.dataset_ratio[1]), len(cls_img_list) - (cls_fold_img_nb * (self.dataset_ratio[0] + self.dataset_ratio[1]))] # train
                ]
                if self.debug:
                    print("\nclass image numbers(test, val, train): {0: 5}/{1: 5}/{2: 5}".format(
                        cls_img_counts[0][1],
                        cls_img_counts[1][1] - cls_img_counts[0][1],
                        cls_img_counts[2][1]
                    ))
                for c, cls_img_count in enumerate(cls_img_counts):
                    for img_idx in range(cls_img_count[0], cls_img_count[1]):
                        target_cls_dir = os.path.join(self.target_dir, self.dataset_type[c], self.filter_eng_class[i])
                        origin_img_path = os.path.join(self.origin_dir, kcls, cls_img_list[img_idx])
                        target_img_path = os.path.join(target_cls_dir, "{0:06}.jpg".format(img_idx))
                        shutil.copy(origin_img_path, target_img_path)
                        if self.debug:
                            if c == 0:
                                eng_class = self.filter_eng_class[i].ljust(20)
                                idx = img_idx
                                count = cls_img_count[1] - 1
                            elif c == 1:
                                eng_class = " ".ljust(20)
                                idx = img_idx - cls_img_count[0]
                                count = cls_img_count[1] - cls_img_count[0] - 1
                            else:
                                eng_class = " ".ljust(20)
                                idx = img_idx
                                count = cls_img_count[1] - 1
                            print("\r{0} {1}: {2: 5}/{3: 5} - copy {4} to {5}".format(
                                    eng_class,
                                    self.dataset_type[c].ljust(5),
                                    idx, count,
                                    origin_img_path.ljust(20), target_img_path
                                ), end="")
                    if cls_img_count[1] > 0 and self.debug:
                        print()
                self.cls_nb_img.append(cls_img_counts)
            return True
        except:
            return False

    def save_data(self):
        self.data_path = os.path.join("data", "bookcover-{}.yaml".format(len(self.filter_eng_class)))
        ldata = []
        test_path = os.path.join(self.target_dir, 'test')
        val_path = os.path.join(self.target_dir, 'val')
        train_path = os.path.join(self.target_dir, 'train')

        ldata.append(f"# Book Cover {len(self.filter_eng_class)} classification dataset\n")
        ldata.append(f"test: {test_path}\n")
        ldata.append(f"val: {val_path}\n")
        ldata.append(f"train: {train_path}\n\n")
        ldata.append(f"# number of class\n")
        ldata.append(f"nc: {len(self.filter_eng_class)}\n\n")
        ldata.append(f"# class name\n")
        ldata.append(f"names: {self.filter_eng_class}")

        write_data(self.data_path, ldata)

        print(Logging.i("data file path: {}".format(self.data_path)))

    def save_dataset_info(self):
        self.rename_dataset_path = os.path.join(self.target_dir, "dataset_info.txt")
        nc = len(self.filter_eng_class)
        origin_dir = self.origin_dir
        target_dir = self.target_dir

        ldata = []
        ldata.append("number of class: {}\n".format(nc))
        ldata.append("original dataset directory: {}\n".format(origin_dir))
        ldata.append("target dataset directory: {}\n\n".format(target_dir))
        ldata.append("Renamed dataset information\n")
        ldata.append("korea class\tenglish class\ttrain\tvalid\ttest\n".format(nc))
        for i, nb_img in enumerate(self.cls_nb_img):
            ldata.append("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                self.filter_kor_class[i],
                self.filter_eng_class[i],
                nb_img[0][1],
                nb_img[1][1],
                nb_img[2][1],
                nb_img[0][1] + nb_img[1][1] + nb_img[2][1]
            ))
        write_data(self.rename_dataset_path, ldata)
        print(Logging.i("Rename dataset information file path: {}".format(self.rename_dataset_path)))
