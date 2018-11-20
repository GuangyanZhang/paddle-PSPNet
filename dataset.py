import os
import pickle

import numpy as np
from PIL import Image
import random

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
        img_pkl_path = root + img_pkl_path
        lab_pkl_path = root + lab_pkl_path

        self.count_data = 2913

        if not(os.path.isfile(img_pkl_path)):
            self.filenames = [image_basename(f)
                for f in os.listdir(self.labels_root) if is_image(f)]
            self.filenames.sort()

            self.input_transform = input_transform
            self.target_transform = target_transform

            dump_img = open(img_pkl_path, 'wb')
            dump_lab = open(lab_pkl_path, 'wb')
            for i in range(self.count_data):
                img, lab = self.getitem(i)
                pickle.dump(img, dump_img, -1)
                pickle.dump(lab, dump_lab, -1)
            dump_img.close()
            dump_lab.close()
        print '[info] loading datasets with saved pkl file.'
        self.index = 0
        load_img = open(img_pkl_path, 'rb')
        load_lab = open(lab_pkl_path, 'rb')
        self.image = []
        self.label = []
        for _ in range(self.count_data):
            self.image.append(pickle.load(load_img))
            self.label.append(pickle.load(load_lab))
        load_img.close()
        load_lab.close()

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

    def random_crop_or_pad(self, img, mask, width, height):
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]

        if img.shape[0] >= height:
            y = random.randint(0, img.shape[0] - height)
            img = img[y:y+height, :]
            mask = mask[y:y+height, :]
        else:
            y = random.randint(0, height - img.shape[0])
            shape = [y, height - img.shape[0] - y]
            img = np.pad(img, [shape, [0, 0], [0, 0]], 'constant')
            mask = np.pad(mask, [shape, [0, 0]], 'constant')

        if img.shape[1] >= width:
            x = random.randint(0, img.shape[1] - width)
            img = img[:, x:x+width]
            mask = mask[:, x:x+width]
        else:
            x = random.randint(0, width - img.shape[1])
            shape = [x, width - img.shape[1] - x]
            img = np.pad(img, [[0, 0], shape, [0, 0]], 'constant')
            mask = np.pad(mask, [[0, 0], shape], 'constant')
        return img, mask

    def get_batch(self, bath_size, width, height):
        img = []
        lab = []
        for _ in range(bath_size):
            if (self.index == self.count_data):
                self.index = 0
            croped_img, croped_mask = \
                    self.random_crop_or_pad(np.array(self.image[self.index]), 
                                            np.array(self.label[self.index]), 
                                            width, 
                                            height)
            img.append(croped_img)
            lab.append(croped_mask)
            self.index += 1
        img = np.array(img).astype('float32')
        lab = np.array(lab).astype('int64')
        return img, lab

    def get_reader(self, width, height):
        def reader():
            for i in range(self.count_data):
                yield self.random_crop_or_pad(np.array(self.image[i]), 
                                            np.array(self.label[i]), 
                                            width, 
                                            height)
        return reader