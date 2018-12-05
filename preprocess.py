# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:28:26 2018

@author: Weiyu_Lee
"""

import os
import numpy as np
import scipy.misc
import random
from tqdm import tqdm

def load_image_data(input_path, label_path, aug_enable=True, nrml_enable=False):
    
    input_data = []
    label_data = []
    image_names = []
    
    input_data_list = os.listdir(input_path)
    
    data_pbar = tqdm(range(len(input_data_list)))
    
    for f_idx in data_pbar:
        f = input_data_list[f_idx]
        
        file_name = os.path.splitext(f)[0]

        curr_input = np.array(scipy.misc.imread(os.path.join(input_path, f)))
        curr_label = np.array(scipy.misc.imread(os.path.join(label_path, f)))

        if len(curr_input.shape) < 3:
            curr_input = np.expand_dims(curr_input, axis=-1)

        if nrml_enable is True:
            curr_input = curr_input / 255.
            
        input_data.append(curr_input)
        label_data.append(curr_label)
        image_names.append(file_name)

        if aug_enable is True:
            # ==================== Flip ====================
            curr_image_flipud = np.flipud(curr_input)
            curr_code_flipud = np.flipud(curr_label)
            
            curr_image_fliplr = np.fliplr(curr_input)
            curr_code_fliplr = np.fliplr(curr_label)

            image_names.append(file_name + "_flipud")                
            image_names.append(file_name + "_fliplr")                                
            
            input_data.append(curr_image_flipud)
            input_data.append(curr_image_fliplr)
            
            label_data.append(curr_code_flipud)
            label_data.append(curr_code_fliplr)
            
            # ==================== Rotate ====================
            curr_image_rot90 = np.rot90(curr_input)
            curr_code_rot90 = np.rot90(curr_label)
    
            curr_image_rot180 = np.rot90(curr_image_rot90)
            curr_code_rot180 = np.rot90(curr_code_rot90)                   

            curr_image_rot270 = np.rot90(curr_image_rot180)
            curr_code_rot270 = np.rot90(curr_code_rot180)                   

            image_names.append(file_name + "_rot90")                
            image_names.append(file_name + "_rot180")                
            image_names.append(file_name + "_rot270")     

            input_data.append(curr_image_rot90)
            input_data.append(curr_image_rot180)                
            input_data.append(curr_image_rot270)
            
            label_data.append(curr_code_rot90)
            label_data.append(curr_code_rot180)
            label_data.append(curr_code_rot270)                
            
            # ==================== Flip ud & Rotate ====================
            curr_image_flipud_rot90 = np.rot90(curr_image_flipud)
            curr_code_flipud_rot90 = np.rot90(curr_code_flipud)
    
            curr_image_flipud_rot180 = np.rot90(curr_image_flipud_rot90)
            curr_code_flipud_rot180 = np.rot90(curr_code_flipud_rot90)                   

            curr_image_flipud_rot270 = np.rot90(curr_image_flipud_rot180)
            curr_code_flipud_rot270 = np.rot90(curr_code_flipud_rot180)                   

            image_names.append(file_name + "_flipud_rot90")                
            image_names.append(file_name + "_flipud_rot180")                
            image_names.append(file_name + "_flipud_rot270")     

            input_data.append(curr_image_flipud_rot90)
            input_data.append(curr_image_flipud_rot180)                
            input_data.append(curr_image_flipud_rot270)
            
            label_data.append(curr_code_flipud_rot90)
            label_data.append(curr_code_flipud_rot180)
            label_data.append(curr_code_flipud_rot270)                

            # ==================== Flip lr & Rotate ====================
            curr_image_fliplr_rot90 = np.rot90(curr_image_fliplr)
            curr_code_fliplr_rot90 = np.rot90(curr_code_fliplr)
    
            curr_image_fliplr_rot180 = np.rot90(curr_image_fliplr_rot90)
            curr_code_fliplr_rot180 = np.rot90(curr_code_fliplr_rot90)                   

            curr_image_fliplr_rot270 = np.rot90(curr_image_fliplr_rot180)
            curr_code_fliplr_rot270 = np.rot90(curr_code_fliplr_rot180)                   

            image_names.append(file_name + "_fliplr_rot90")                
            image_names.append(file_name + "_fliplr_rot180")                
            image_names.append(file_name + "_fliplr_rot270")     

            input_data.append(curr_image_fliplr_rot90)
            input_data.append(curr_image_fliplr_rot180)                
            input_data.append(curr_image_fliplr_rot270)
            
            label_data.append(curr_code_fliplr_rot90)
            label_data.append(curr_code_fliplr_rot180)
            label_data.append(curr_code_fliplr_rot270) 
            
    print("[load_image_data] input_data shape: {}".format(np.array(input_data).shape))
    
    return input_data, label_data, image_names

def split_image(input, label, image_name, sub_size):
    
    height, width, _ = label.shape
    
    output_image = []
    output_label = []
    output_meta_data = []
    
    for h in range(0, height-sub_size, sub_size):
        for w in range(0, width-sub_size, sub_size):    
            
            curr_input = input[h:h+sub_size, w:w+sub_size]
            curr_label = label[h:h+sub_size, w:w+sub_size]
            
            output_image.append(curr_input)            
            
            if curr_label[:, :, 0].sum() > 0: 
                output_meta_data.append((image_name, [h,w], "regular_grid", "abnormal"))        
                output_label.append(1)
            else:
                output_meta_data.append((image_name, [h,w], "regular_grid", "normal"))                    
                output_label.append(0)

    output_image = np.array(output_image)
    output_label = np.array(output_label)
    
    #print("[split_image] normal: {}, abnormal: {}".format(output_label.shape[0] - output_label.sum(), output_label.sum()))                                
    #print("[split_image] output_label shape: {}".format(output_label.shape))
    
    return output_image, output_label, output_meta_data

def random_crop_image(input, label, image_name, sub_size, crop_num):
    
    height, width, _ = label.shape
    output_image = []
    output_label = []
    output_meta_data = []
    
    count = 0
    
    while (count < crop_num):
        h = random.randint(0, height-sub_size)
        w = random.randint(0, width-sub_size)
        
        curr_input = input[h:h+sub_size, w:w+sub_size]
        curr_label = label[h:h+sub_size, w:w+sub_size]   

        # abnormal            
        if curr_label[:, :, 0].sum() > 0: 
            output_label.append(1)        
            output_image.append(curr_input)
            output_meta_data.append((image_name, [h,w], "random_grid", "abnormal"))
           
        # normal
        else:
            output_label.append(0)        
            output_image.append(curr_input)
            output_meta_data.append((image_name, [h,w], "random_grid", "normal"))

        count += 1

    output_image = np.array(output_image)
    output_label = np.array(output_label)

    #print("[random_crop_image] normal: {}, abnormal: {}".format(output_label.shape[0] - output_label.sum(), output_label.sum()))              
    #print("[random_crop_image] output_label shape: {}".format(np.array(output_label).shape))
    
    return output_image, output_label, output_meta_data

def main_process():
    
    data_type = 'train'
    SUB_SIZE = 200
    CROP_NUM = 512
    
    input_path = '/data/wei/dataset/MDetection/Stem_cell/' + data_type + '_data/data_preprocessed/'
    label_path = '/data/wei/dataset/MDetection/Stem_cell/' + data_type + '_data/label/'
    
    # Define containers
    normal_data = []
    normal_label = []
    abnormal_data = []
    abnormal_label = []
    
    # Load all the images and apply augmentation process
    input_data, label_data, image_names = load_image_data(input_path, label_path)
    
    # Crop images
    for idx in range(len(input_data)):
        
        # Cut images into patches
        if data_type == 'train':
            patch_image, patch_label, patch_meta = random_crop_image(input_data[idx], label_data[idx], image_names[idx], SUB_SIZE, CROP_NUM)
        elif data_type == 'test':
            patch_image, patch_label, patch_meta = split_image(input_data[idx], label_data[idx], image_names[idx], SUB_SIZE)
        
        # Classifiy the data
        for p_idx in range(patch_label.shape[0]):
            if patch_label[p_idx] == 1:    
                abnormal_data.append(patch_image[p_idx])
                abnormal_label.append(patch_label[p_idx])
            elif patch_label[p_idx] == 0:    
                normal_data.append(patch_image[p_idx])
                normal_label.append(patch_label[p_idx])
    
    print("[main_process] normal_data: {}, abnormal_data: {}".format(len(normal_data), len(abnormal_data)))
    
    np.save('/data/wei/dataset/MDetection/Stem_cell/' + data_type + '_data/normal_data_.npy', {'data':normal_data, 'label':normal_label})
    np.save('/data/wei/dataset/MDetection/Stem_cell/' + data_type + '_data/abnormal_data_.npy', {'data':abnormal_data, 'label':abnormal_label})
    
if __name__ == '__main__':

    main_process()    