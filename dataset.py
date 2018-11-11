import os
import pickle

import numpy as np
from PIL import Image

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, basename + extension)

def image_basename(filename,):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC:

    def __init__(self, root, img_pkl_path = 'img.pkl', lab_pkl_path = 'lab.pkl', input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'classes')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.img = []
        self.lab = []
        if os.path.isfile(img_pkl_path):
            print '[info] loading datasets with saved pkl file.'
            load_img = open(img_pkl_path, 'rb')
            load_lab = open(lab_pkl_path, 'rb')
            for _ in range(self.len()):
                self.img.append(pickle.load(load_img))
                self.lab.append(pickle.load(load_lab))
        else:
            dump_img = open(img_pkl_path, 'wb')
            dump_lab = open(lab_pkl_path, 'wb')
            for i in range(self.len()):
                img, lab = self.getitem(i)
                pickle.dump(img, dump_img, -1)
                pickle.dump(lab, dump_lab, -1)
                if (i%10 == 0):
                    print(i/10)

    def getitem(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def len(self):
        return len(self.filenames)
