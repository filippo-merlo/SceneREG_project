from config_powerpaint import *
from utils_powerpaint import *
from dataset import *

if __name__ == '__main__':
    dataset = Dataset(dataset_path = dataset_path)
    data = dataset.data
    n = 2
    generate_new_images(data, n)


