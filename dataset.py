from torch.utils.data.dataset import Dataset
from torchvision import transforms

import os
import random
from PIL import Image
import xml.etree.ElementTree as ET
import pickle

class VOCDataset(Dataset):
    def __init__(self):
        super(VOCDataset, self).__init__()
        self.image_path = "./data/VOCdevkit/VOC2012/JPEGImages"
        self.annotation_path = "./data/VOCdevkit/VOC2012/Annotations"
        self.cache_file = "./data/saved_dataset.pkl"

        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        if os.path.exists(self.cache_file):
            self.dataset = self._load_cached_dataset()
        else:
            self.image_names = [
                img.split(".")[0] for img in os.listdir(self.image_path)
            ]
            self.dataset = self._build_dataset()

    def _load_cached_dataset(self):
        with open(self.cache_file, 'rb') as fp:
            res = pickle.load(fp)
        return res

    def _build_dataset(self):
        res = []
        for image_name in self.image_names:
            # xml file path for the image
            annotation_path = self.annotation_path + "/" + image_name + ".xml"

            tree = ET.parse(annotation_path)
            root = tree.getroot()

            objects = root.findall('object')
            for o in objects:
                box_info = o.find('bndbox')
                try:
                    xmin = int(box_info.find('xmin').text)
                    xmax = int(box_info.find('xmax').text)
                    ymin = int(box_info.find('ymin').text)
                    ymax = int(box_info.find('ymax').text)

                    # append the image name along with box info
                    box = (xmin, ymin, xmax, ymax)
                    res.append((image_name, box))

                except:
                    # ignore files with float values for pixel coords
                    pass

        # save in cache
        with open(self.cache_file, 'wb') as fp:
            pickle.dump(res, fp)

        return res

    def random_test(self):
        return self.__getitem__(random.randint(0, len(self.dataset)))

    def __getitem__(self, index):
        img_name, box = self.dataset[index]
        img_path = os.path.join(self.image_path, img_name + ".jpg")

        # load PIL image and apply transformations
        img = Image.open(img_path)
        width, height = img.size

        # transform the image
        img = self.transformations(img)
        # transform the bounding box corrds a/c to resize
        x1 = int(box[0]/width * 224)
        y1 = int(box[1]/height * 224)
        x2 = int(box[2]/width * 224)
        y2 = int(box[3]/height * 224)

        return img, (x1, y1, x2, y2)


    def __len__(self):
        return len(self.dataset)
