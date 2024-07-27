from config import *
from utils import *
from dataset import *

if __name__ == '__main__':
    dataset = Dataset(dataset_path = dataset_path)
    data = dataset.data
    n = 5
    generate_new_images(data, n)


