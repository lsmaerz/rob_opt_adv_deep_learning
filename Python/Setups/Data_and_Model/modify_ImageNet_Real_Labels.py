'''
@author = Lars Steffen März
This file prepends the ground truth class labels to the image file names.
It follows the validation dataset ground truth from the ImageNet dev kit.
'''

import os
def attach_text_to_filenames(txt_file_path, dir_path):
    with open(txt_file_path) as f:
        text = f.readlines()
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files.sort()
    for i, filename in enumerate(files):
        if i >= len(text):
            break
        filepath = os.path.join(dir_path, filename)
        new_filename = str(int(text[int(filename.split("_")[2].split(".")[0])-1])-1) + '_' + filename
        new_filepath = os.path.join(dir_path, new_filename)
        print(new_filepath)
        os.rename(filepath, new_filepath)

if __name__=="__main__":
    attach_text_to_filenames('./Data/ImageNet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt', './Data/ImageNet/images')