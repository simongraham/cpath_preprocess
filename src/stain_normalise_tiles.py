import os
import cv2
import glob
from multiprocessing import Pool
from functools import partial
from dataloader import save_tiles_multi_cpu
from preprocessing.stainnorm.stain_extractor import get_stain_matrix
from preprocessing.stainnorm.utils import *

def single_file_run(source,
                    target_img,
                    sn_method='vahadane',
                    output_dir='patches'):
    
    basename = os.path.basename(source)
    source_img = cv2.imread(source)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB) 

    # Get the stain matrices of source and target images
    if sn_method == 'vahadane':
        stain_matrix_target = get_stain_matrix(target_img, 'vahadane')
        stain_matrix_source = get_stain_matrix(source_img, 'vahadane')
    elif sn_method == 'mackenko':
        stain_matrix_target = get_stain_matrix(target_img, 'mackenko')
        stain_matrix_source = get_stain_matrix(source_img, 'mackenko')
    
    if sn_method == 'vahadane' or sn_method == 'mackenko':
        # Get stain concentrations
        target_concentrations = get_concentrations(target_img, stain_matrix_target)
        maxC_target = np.percentile(target_concentrations, 99, axis=0).reshape((1, 2))
        stain_matrix_target_RGB = convert_OD_to_RGB(stain_matrix_target)  # useful to visualize.
        
        source_concentrations = get_concentrations(source_img, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (maxC_target / maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target))

        normed_img = tmp.reshape(source_img.shape).astype(np.uint8)
        normed_img = cv2.cvtColor(normed_img, cv2.COLOR_BGR2RGB) 
        
        # Save image
        cv2.imwrite(os.path.join(output_dir, basename), normed_img)

    elif sn_method == 'reinhard':
        I1_t, I2_t, I3_t = lab_split(target_img)
        m1_t, sd1_t = cv2.meanStdDev(I1_t)
        m2_t, sd2_t = cv2.meanStdDev(I2_t)
        m3_t, sd3_t = cv2.meanStdDev(I3_t)
        target_means = m1_t, m2_t, m3_t
        target_stds = sd1_t, sd2_t, sd3_t

        I1_s, I2_s, I3_s = lab_split(source_img)
        m1_s, sd1_s = cv2.meanStdDev(I1_s)
        m2_s, sd2_s = cv2.meanStdDev(I2_s)
        m3_s, sd3_s = cv2.meanStdDev(I3_s)
        source_means = m1_s, m2_s, m3_s
        source_stds = sd1_s, sd2_s, sd3_s

        norm1 = ((I1_s - source_means[0]) * (target_stds[0] / source_stds[0])) + target_means[0]
        norm2 = ((I2_s - source_means[1]) * (target_stds[1] / source_stds[1])) + target_means[1]
        norm3 = ((I3_s - source_means[2]) * (target_stds[2] / source_stds[2])) + target_means[2]
        
        normed_img = merge_back(norm1, norm2, norm3)
        normed_img = cv2.cvtColor(normed_img, cv2.COLOR_BGR2RGB)

        # Save image
        cv2.imwrite(os.path.join(output_dir, basename), normed_img)

def stain_normalise_tiles(source_path,
        target=None,
        sn_method='vahadane',
        output_dir='patches',
        file_name_pattern = 'jpg',
        num_cpu=4
        ):
    '''
    Include Description
    '''

    assert sn_method == 'vahadane' or sn_method == 'mackenko' or sn_method == 'reinhard',\
    "Choose from 'vahadane', 'mackenko' or 'reinhard'."
    
    if output_dir is None:
        if os.path.isfile(source_path):
            dir_path, _ = os.path.split(source_path)
            output_dir = os.path.join(dir_path, '..', 'sn_patches')
        elif os.path.isdir(source_path):
            output_dir = os.path.join(source_path, '..', 'sn_patches')
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if target is None:
        # Read the pre-defined target image in repo for stain norm
        target_path = 'target.png'
        target_img = cv2.imread(target_path)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    else:
        # Read the target image for stain norm
        target_path = target
        target_img = cv2.imread(target_path)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
    if os.path.isdir(source_path):
        files_all = sorted(glob.glob(os.path.join(source_path, file_name_pattern)))
        with Pool(num_cpu) as p:
            p.map(partial(single_file_run,
                        target_img=target_img,
                        sn_method=sn_method,
                        output_dir=output_dir),
                        files_all)

    if os.path.isfile(source_path):
        single_file_run(source=source,
                        target_img=target_img,
                        sn_method=sn_method,
                        output_dir=output_dir)
    