"""
This file preprocess the raw MNIH Massachusetts Roads Dataset and extract the images into small patches
"""


# Built-in
import os
import glob
import pathlib
import fnmatch

# Libs
import numpy as np
from tqdm import tqdm

# Own modules
import sys
sys.path.append("/Users/azanchetta/OneDrive - The Alan Turing Institute/Research/projects/testingML/trials")
print(sys.path)
from data import data_utils
from mrs_utils import misc_utils

# Settings
DATA_DIR = '/Users/azanchetta/OneDrive - The Alan Turing Institute/Research/projects/testingML/trials/data/mnih/'
SPLITS = ['train', 'valid'] # test set will be grabbed by get_images() and processed during testing
# MODES = os.listdir(os.path.join(DATA_DIR, SPLITS[0])) # sat (input), map (target)   - ORIGINAL
MODES = [('map','*.tif'),('sat','*.tiff')]
MEAN = (0.4251811, 0.42812928, 0.39143909)
STD = (0.22423858, 0.21664895, 0.22102307)

def patch_tile(rgb_file, gt_file, patch_size, pad, overlap):
    """
    Extract the given rgb and gt tiles into patches
    :param rgb_file: path to the rgb file
    :param gt_file: path to the gt file
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return: rgb and gt patches as well as coordinates
    """
    rgb = misc_utils.load_file(rgb_file)
    gt = misc_utils.load_file(gt_file)
    # np.testing.assert_array_equal(rgb.shape[:2], gt.shape)
    np.testing.assert_array_equal(rgb.shape[:2], gt.shape[:2]) ### ANNA's EDITS HERE but probably not correct
    # the gt files ('mnih/train/map') have 3 bands, not 1 (apparently were supposed to be?), the info is in the 1st one, other are only 0s
    # for some time I had this script work putting `gt.shape[:0]`
    grid_list = data_utils.make_grid(np.array(rgb.shape[:2]) + 2 * pad, patch_size, overlap)
    if pad > 0:
        rgb = data_utils.pad_image(rgb, pad)
        gt = data_utils.pad_image(gt, pad)
    for y, x in grid_list:
        rgb_patch = data_utils.crop_image(rgb, y, x, patch_size[0], patch_size[1])
        gt_patch = data_utils.crop_image(gt, y, x, patch_size[0], patch_size[1])
        yield rgb_patch, gt_patch, y, x


def patch_mnih(data_dir, save_dir, patch_size, pad, overlap):
    """
    Preprocess the standard mnih dataset
    :param data_dir: path to the original mnih dataset
    :param save_dir: directory to save the extracted patches
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return:
    """
    
    for dataset in tqdm(SPLITS, desc='Train-valid split'):
        for MY_MODE in MODES:
            modes_folder_name = os.path.join(DATA_DIR, dataset,MY_MODE[0])
            
            print("****\n****\modes_folder_name is",modes_folder_name)
        
            
            file_names_list = glob.glob(modes_folder_name + "/" + MY_MODE[1])
            
            #print('- - - - \n- - - -\nfile names:',file_names_list)
            
            
            filenames_with_path_extension_stripped = [
                # fname.split('.')[0] for fname in os.listdir(os.path.join(DATA_DIR, dataset, MODES[0]))
                
                file_name.split('.')[0] for file_name in file_names_list  #### THIS WORKS!!!
                
            ]
            # create folders and files
            patch_dir = os.path.join(save_dir, 'patches')
            misc_utils.make_dir_if_not_exist(patch_dir)
            record_file = open(os.path.join(save_dir, 'file_list_{}.txt'.format(dataset)), 'w+')

            # get rgb and gt files
            for filename_with_path_extension_stripped in tqdm(filenames_with_path_extension_stripped, desc='File-wise'):
                print("filename is", filename_with_path_extension_stripped, '***\n***')
                filename_extension_stripped = pathlib.Path(filename_with_path_extension_stripped).stem
                print("image name for preprocess is:", filename_extension_stripped)
                rgb_filename = os.path.join(DATA_DIR, dataset, 'sat', filename_extension_stripped +'.tiff')
                print('\n----\nrgb filename is:', rgb_filename)
                gt_filename = os.path.join(DATA_DIR, dataset, 'map', filename_extension_stripped+'.tif')
                print('\n----\ngt filename is:', gt_filename, '\n----\n')
                for rgb_patch, gt_patch, y, x in patch_tile(rgb_filename, gt_filename, patch_size, pad, overlap):
                    rgb_patchname = '{}_y{}x{}.jpg'.format(filename_with_path_extension_stripped, int(y), int(x))
                    gt_patchname = '{}_y{}x{}.png'.format(filename_with_path_extension_stripped, int(y), int(x))
                    misc_utils.save_file(os.path.join(patch_dir, rgb_patchname), rgb_patch.astype(np.uint8))
                    misc_utils.save_file(os.path.join(patch_dir, gt_patchname), (gt_patch/255).astype(np.uint8))
                    record_file.write('{} {}\n'.format(rgb_patchname, gt_patchname))
            record_file.close()


def get_images(data_dir=DATA_DIR, dataset='test'):
    """
    Stand-alone function to be used in evaluate.py.
    :param data_dir
    :param dataset: name of the dataset/split
    """    
    rgb_files = []
    gt_files = []
    file_list = os.listdir(os.path.join(data_dir, dataset, 'map'))
    for fname in file_list:
        gt_files.append(os.path.join(data_dir, dataset, 'map' ,fname))
        rgb_files.append(os.path.join(data_dir, dataset, 'sat' ,fname.replace('tif', 'tiff')))
    return rgb_files, gt_files


def get_stats(img_dir):
    from data import data_utils
    from glob import glob
    rgb_imgs = []
    for set_name in ['train', 'valid', 'test']:
        rgb_imgs.extend(glob(os.path.join(img_dir, set_name, 'sat', '*.tiff')))
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    print('Mean: {}'.format(ds_mean))
    print('Std: {}'.format(ds_std))


if __name__ == '__main__':
    ps = 512
    pd = 0
    ol = 0
    save_dir = r'/Users/azanchetta/OneDrive - The Alan Turing Institute/Research/projects/testingML/trials/output/mnih/processed_mnih/'
    misc_utils.make_dir_if_not_exist(save_dir)
    patch_mnih(data_dir=DATA_DIR,
               save_dir=save_dir,
               patch_size=(ps, ps),
               pad=pd,
               overlap=ol)
