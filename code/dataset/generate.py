from config import *
from utils import *
from dataset import *

if __name__ == '__main__':
    dataset = Dataset(dataset_path = dataset_path)
    data = dataset.data
    n = 3
    generate_new_image(data, n)
    #try_things(data)

