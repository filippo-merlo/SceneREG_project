from config import *
from utils import *
from dataset import *

if __name__ == '__main__':
    dataset = Dataset(dataset_path = coco_ann_path)
    data = dataset.data
    generate_new_image(data)

