import os
import shutil

# Define the source and target directories
source_dir = '/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26/images'
target_dir = '/mnt/cimec-storage6/users/filippo.merlo/ade20k_adapted'

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def copy_files(src, dst):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith('.jpg'):
                # Get the category name which is the parent directory of the current directory
                category = os.path.basename(os.path.dirname(root))
                target_path = os.path.join(dst, category)
                ensure_dir_exists(target_path)
                shutil.move(os.path.join(root, file), os.path.join(target_path, file))

def main():
    # Directories to process
    dirs_to_process = {
        'training': 'train',
        'validation': 'val'
    }

    for src_subdir, dst_subdir in dirs_to_process.items():
        src_path = os.path.join(source_dir, src_subdir)
        dst_path = os.path.join(target_dir, dst_subdir)
        ensure_dir_exists(dst_path)
        copy_files(src_path, dst_path)

if __name__ == '__main__':
    main()