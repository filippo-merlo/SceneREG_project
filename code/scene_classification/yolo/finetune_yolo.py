from ultralytics import YOLO
import os

# import YOLO model

data_path = '/mnt/cimec-storage6/users/filippo.merlo/ade20k_adapted'
n_epochs = 100
bs = 16
#gpu_id = 0
verbose = True
rng = 0
validate = True
model = YOLO('yolov8n-cls.pt')

# Specify the save directory for training runs
save_dir = '/mnt/cimec-storage6/users/filippo.merlo/yolo'
os.makedirs(save_dir, exist_ok=True)

# Train the model

results = model.train(
    data=data_path,
    epochs=n_epochs,
    batch=bs,
    #device=gpu_id,
    verbose=verbose,
    seed=rng,
    val=validate,
    save_dir=save_dir
)

# Validate the model
metrics = model.val() # no arguments needed, dataset and settings remembered
metrics.top1 # top1 accuracy
metrics.top5 # top5 accuracy



