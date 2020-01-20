import os
import cv2
from multiprocessing import Pool
from functools import partial
from dataloader import save_tiles_multi_cpu


def extract_patches(wsi_path,
        output_dir='patches',
        file_name_pattern='*.svs',
        num_cpu=4,
        tile_objective_value=20,
        tile_read_size_w=5000,
        tile_read_size_h=5000,
        nr_tiles=None,
        tiss_level=4,
        use_tiss_mask=True,
        tiss_cutoff = 0.1
        ):
 
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if output_dir is None:
        if os.path.isfile(wsi_path):
            dir_path, _ = os.path.split(wsi_path)
            output_dir = os.path.join(dir_path, '..', 'patches')
        elif os.path.isdir(wsi_path):
            output_dir = os.path.join(wsi_path, '..', 'patches')

    opts = {
        'output_dir': output_dir,
        'wsi_input': wsi_path,
    }
    
    save_tiles_multi_cpu.run(opts_in=opts,
                            file_name_pattern=file_name_pattern,
                            num_cpu=num_cpu,
                            tile_objective_value=tile_objective_value,
                            tile_read_size_w=tile_read_size_w,
                            tile_read_size_h=tile_read_size_h,
                            nr_tiles=nr_tiles,
                            use_tiss_mask=use_tiss_mask,
                            tiss_level=tiss_level,
                            tiss_cutoff=tiss_cutoff
                            )

    