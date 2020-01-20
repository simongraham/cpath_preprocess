import os
import numpy as np
import yaml
import math
from PIL import Image
import pandas as pd
import cv2
import skimage
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.filters import rank, threshold_otsu
from skimage.transform import resize
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import (binary_dilation, binary_erosion,
                                      binary_fill_holes, binary_closing)

if os.name == 'nt':
    os.environ['PATH'] = "C:\\tools\\openslide\\openslide-win64-20171122\\bin" + ";" + os.environ['PATH']
import openslide as openslide


class TissueGenerator:
    def __init__(self,
                 input_dir=os.getcwd(),
                 file_name='Test_file.svs',
                 output_dir=os.path.join(os.getcwd(), 'tissue'),
                 tiss_level=4):

        self.input_dir = input_dir
        self.file_name = os.path.basename(file_name)
        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, self.file_name)
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)

        self.openslide_obj = openslide.OpenSlide(filename=os.path.join(self.input_dir, self.file_name))
        self.tiss_level = tiss_level
        self.level_count = self.openslide_obj.level_count
        self.level_dimensions = self.openslide_obj.level_dimensions
        self.level_downsamples = self.openslide_obj.level_downsamples

    def read_region(self, start_w, start_h, end_w, end_h, level):
        openslide_obj = self.openslide_obj
        im_region = openslide_obj.read_region([start_w, start_h], level, [end_w - start_w, end_h - start_h])

        return im_region

    def load_wsi(self):
        openslide_obj = self.openslide_obj
        if self.level_count-1 < self.tiss_level:
            self.tiss_level = self.level_count-1
        slide_dimension = openslide_obj.level_dimensions[self.tiss_level]
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]

        output_dir = self.output_dir
        im = self.read_region(0, 0, slide_w, slide_h, level=self.tiss_level)

        temp = np.array(im)
        temp = temp[:, :, 0:3]
        self.shape_x = temp.shape[1]
        self.shape_y = temp.shape[0]
        im = Image.fromarray(temp)
        self.im = im
        self.im_resize =  self.im.copy()

    def stain_entropy_otsu(self):
        im_copy = self.im_resize.copy()
        hed = skimage.color.rgb2hed(im_copy)  # convert colour space
        hed = (hed * 255).astype(np.uint8)
        h = hed[:, :, 0]
        e = hed[:, :, 1]
        d = hed[:, :, 2]
        selem = disk(4)  # structuring element
        # calculate entropy for each colour channel
        h_entropy = rank.entropy(h, selem)
        e_entropy = rank.entropy(e, selem)
        d_entropy = rank.entropy(d, selem)
        entropy = np.sum([h_entropy, e_entropy], axis=0) - d_entropy
        # otsu threshold
        threshold_global_otsu = threshold_otsu(entropy)
        self.otsu = entropy > threshold_global_otsu

    def morphology(self):
        # Join together large groups of small components ('salt')
        radius = int(8)
        selem = disk(radius)
        dilate = binary_dilation(self.otsu, selem)

        radius = int(16)
        selem = disk(radius)
        erode = binary_erosion(dilate, selem)

        rm_holes = remove_small_holes(
            erode,
            area_threshold=int(40)**2,
            connectivity=1,
        )

        closing = binary_closing(rm_holes, selem)
        rm_objs = remove_small_objects(
            closing,
            min_size=int(120)**2,
            connectivity=1,
        )

        dilate = binary_dilation(rm_objs, selem)

        rm_holes = remove_small_holes(
            dilate,
            area_threshold=int(40)**2,
            connectivity=1,
        )

        self.mask = ndimage.binary_fill_holes(rm_holes)
        self.mask = self.mask.astype('uint8')

    def save_file(self):
        im_save_name = 'Tissue.png'
        cv2.imwrite(os.path.join(self.output_dir, im_save_name), self.mask*255)




