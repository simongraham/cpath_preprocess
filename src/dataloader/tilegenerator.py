import os
import numpy as np
import yaml
import math
from PIL import Image
import pandas as pd
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


class TileGenerator:
    def __init__(self,
                 input_dir=os.getcwd(),
                 file_name='Test_file.svs',
                 output_dir=os.path.join(os.getcwd(), 'tiles'),
                 tile_objective_value=20,
                 tile_read_size_w=3000,
                 tile_read_size_h=3000,
                 nr_tiles=None,
                 tiss_level=4,
                 use_tiss_mask=True,
                 tiss_cutoff=0.1):

        self.input_dir = input_dir
        self.file_name = os.path.basename(file_name)
        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, self.file_name)
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)

        self.openslide_obj = openslide.OpenSlide(filename=os.path.join(self.input_dir, self.file_name))
        self.tile_objective_value = np.int(tile_objective_value)
        self.tile_read_size = np.array([tile_read_size_w, tile_read_size_h])
        self.objective_power = np.int(self.openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        self.level_count = self.openslide_obj.level_count
        self.nr_tiles = nr_tiles
        self.use_tiss_mask = use_tiss_mask
        self.tiss_level = tiss_level
        self.tiss_cutoff = tiss_cutoff
        self.level_dimensions = self.openslide_obj.level_dimensions
        self.level_downsamples = self.openslide_obj.level_downsamples

    def read_region(self, start_w, start_h, end_w, end_h, level=0):
        openslide_obj = self.openslide_obj
        im_region = openslide_obj.read_region([start_w, start_h], level, [end_w - start_w, end_h - start_h])

        return im_region

    def generate_tiles(self):
        openslide_obj = self.openslide_obj
        tile_objective_value = self.tile_objective_value
        tile_read_size = self.tile_read_size

        if self.use_tiss_mask:
            ds_factor = self.level_downsamples[self.tiss_level]

        if self.objective_power == 0:
            self.objective_power = np.int(openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

        rescale = np.int(self.objective_power / tile_objective_value)
        openslide_read_size = np.multiply(tile_read_size, rescale)
        slide_dimension = openslide_obj.level_dimensions[0]
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]
        tile_h = openslide_read_size[0]
        tile_w = openslide_read_size[1]

        iter_tot = 0
        output_dir = self.output_dir
        data = []

        if self.nr_tiles == None:
            for h in range(int(math.ceil((slide_h - tile_h) / tile_h + 1))):
                for w in range(int(math.ceil((slide_w - tile_w) / tile_w + 1))):
                    start_h = h * tile_h
                    end_h = (h * tile_h) + tile_h
                    start_w = w * tile_w
                    end_w = (w * tile_w) + tile_w
                    if end_h > slide_h:
                        end_h = slide_h

                    if end_w > slide_w:
                        end_w = slide_w
                    #
                    if self.use_tiss_mask:
                        tiss = self.mask[int(start_h/ds_factor):int(start_h/ds_factor)+int(openslide_read_size[1]/ds_factor), int(start_w/ds_factor):int(start_w/ds_factor)+int(openslide_read_size[0]/ds_factor)]
                        tiss_frac = np.sum(tiss)/np.size(tiss)
                    else:
                        tiss_frac = 1

                    if tiss_frac > self.tiss_cutoff:

                        im = self.read_region(start_w, start_h, end_w, end_h)
                        format_str = 'Tile%d:  start_w:%d, end_w:%d, start_h:%d, end_h:%d, width:%d, height:%d'

                        print(format_str % (
                            iter_tot, start_w, end_w, start_h, end_h, end_w - start_w, end_h - start_h), flush=True)
                        temp = np.array(im)
                        temp = temp[:, :, 0:3]
                        im = Image.fromarray(temp)
                        if rescale != 1:
                            im = im.resize(size=[np.int((end_w - start_w) / rescale), np.int((end_h - start_h) / rescale)],
                                           resample=Image.BICUBIC)

                        img_save_name = 'Tile' + '_' \
                                        + str(tile_objective_value) + '_' \
                                        + str(int(start_w/rescale)) + '_' \
                                        + str(int(start_h/rescale))\
                                        + '.jpg'

                        im.save(os.path.join(output_dir, img_save_name), format='JPEG')
                        data.append([iter_tot, img_save_name, start_w, end_w, start_h, end_h, im.size[0], im.size[1]])
                        iter_tot += 1

        else:
            for i in range(self.nr_tiles):
                condition = 0
                while condition == 0:
                    h = np.random.randint(0,int(math.ceil((slide_h - tile_h) / tile_h + 1)))
                    w = np.random.randint(0,int(math.ceil((slide_w - tile_w) / tile_w + 1)))

                    start_h = h * tile_h
                    end_h = (h * tile_h) + tile_h
                    start_w = w * tile_w
                    end_w = (w * tile_w) + tile_w
                    if end_h > slide_h:
                        end_h = slide_h

                    if end_w > slide_w:
                        end_w = slide_w
                    #
                    if self.use_tiss_mask:
                        tiss = self.mask[int(start_h/ds_factor):int(start_h/ds_factor)+int(openslide_read_size[1]/ds_factor), int(start_w/ds_factor):int(start_w/ds_factor)+int(openslide_read_size[0]/ds_factor)]
                        tiss_frac = np.sum(tiss)/np.size(tiss)
                    else:
                        tiss_frac = 1

                    if tiss_frac > self.tiss_cutoff:

                        im = self.read_region(start_w, start_h, end_w, end_h)
                        format_str = 'Tile%d:  start_w:%d, end_w:%d, start_h:%d, end_h:%d, width:%d, height:%d'

                        print(format_str % (
                            iter_tot, start_w, end_w, start_h, end_h, end_w - start_w, end_h - start_h), flush=True)
                        temp = np.array(im)
                        temp = temp[:, :, 0:3]
                        im = Image.fromarray(temp)
                        if rescale != 1:
                            im = im.resize(size=[np.int((end_w - start_w) / rescale), np.int((end_h - start_h) / rescale)],
                                            resample=Image.BICUBIC)

                        img_save_name = 'Tile' + '_' \
                                        + str(tile_objective_value) + '_' \
                                        + str(int(start_w/rescale)) + '_' \
                                        + str(int(start_h/rescale))\
                                        + '.jpg'
 
                        im.save(os.path.join(output_dir, img_save_name), format='JPEG')
                        data.append([iter_tot, img_save_name, start_w, end_w, start_h, end_h, im.size[0], im.size[1]])
                        iter_tot += 1
                        condition = 1


        df = pd.DataFrame(data,
                          columns=['iter', 'Tile_Name', 'start_w', 'end_w', 'start_h', 'end_h', 'size_w', 'size_h'])
        df.to_csv(os.path.join(output_dir, 'Output.csv'), index=False)

    def slide_thumbnail(self):
        openslide_obj = self.openslide_obj
        tile_objective_value = self.tile_objective_value
        output_dir = self.output_dir
        file_name = self.file_name

        if self.objective_power == 0:
            self.objective_power = np.int(openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

        _, file_type = os.path.splitext(file_name)
        rescale = np.int(self.objective_power / tile_objective_value)
        slide_dimension = openslide_obj.level_dimensions[0]
        slide_dimension_20x = np.array(slide_dimension) / rescale
        thumb = openslide_obj.get_thumbnail((int(slide_dimension_20x[0] / 16), int(slide_dimension_20x[1]/16)))
        thumb.save(os.path.join(output_dir, 'SlideThumb.jpg'), format='JPEG')

    def param(self, save_mode=True, output_dir=None, output_name=None):
        input_dir = self.input_dir
        if output_dir is None:
            output_dir = self.output_dir
        if output_name is None:
            output_name = 'param.yaml'
        if self.objective_power == 0:
            self.objective_power = np.int(self.openslide_obj.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        objective_power = self.objective_power
        slide_dimension = self.openslide_obj.level_dimensions[0]
        tile_objective_value = self.tile_objective_value
        rescale = np.int(objective_power / tile_objective_value)
        filename = self.file_name
        tile_read_size = self.tile_read_size
        level_count = self.level_count
        level_dimensions = self.level_dimensions
        level_downsamples = self.level_downsamples

        param = {
            'input_dir':input_dir,
            'output_dir': output_dir,
            'objective_power': objective_power,
            'slide_dimension': slide_dimension,
            'rescale': rescale,
            'tile_objective_value': tile_objective_value,
            'filename': filename,
            'tile_read_size': tile_read_size.tolist(),
            'level_count': level_count,
            'level_dimensions': level_dimensions,
            'level_downsamples': level_downsamples
        }
        if save_mode:
            with open(os.path.join(output_dir, output_name), 'w') as yaml_file:
                yaml.dump(param, yaml_file)
        else:
            return param

    def load_ds_wsi(self):
        openslide_obj = self.openslide_obj
        if self.level_count-1 < self.tiss_level:
            self.tiss_level = self.level_count-1
        slide_dimension = openslide_obj.level_dimensions[self.tiss_level]
        slide_h = slide_dimension[1]
        slide_w = slide_dimension[0]

        im = self.read_region(0, 0, slide_w, slide_h, level=self.tiss_level)

        temp = np.array(im)
        temp = temp[:, :, 0:3]
        im = Image.fromarray(temp)
        self.ds_im = im

    def stain_entropy_otsu(self):
        im_copy = self.ds_im.copy()
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


