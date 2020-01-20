from dataloader import tilegenerator
import os
import glob
from multiprocessing import Pool
from functools import partial


def tile_all_process(obj, use_tiss_mask):
    if use_tiss_mask:
        obj.load_ds_wsi()
        obj.stain_entropy_otsu()
        obj.morphology()
  
    obj.generate_tiles()
    obj.slide_thumbnail()
    obj.param()


def single_file_run(file_name, output_dir, input_dir, tile_objective_value, tile_read_size_w, tile_read_size_h, nr_tiles, use_tiss_mask, tiss_level, tiss_cutoff):
    print('Extracting patches from ' + os.path.basename(file_name), flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type == '.svs' or file_type == '.ndpi' or file_type == '.mrxs' or file_type == 'tif' or file_type == 'tiff':
        tile_obj = tilegenerator.TileGenerator(input_dir=input_dir,
                                               file_name=file_name,
                                               output_dir=output_dir,
                                               tile_objective_value=tile_objective_value,
                                               tile_read_size_w=tile_read_size_w,
                                               tile_read_size_h=tile_read_size_h,
                                               nr_tiles=nr_tiles,
                                               use_tiss_mask=use_tiss_mask,
                                               tiss_level=tiss_level,
                                               tiss_cutoff=tiss_cutoff)
        tile_all_process(obj=tile_obj, use_tiss_mask=True)


def run(opts_in,
        file_name_pattern, num_cpu, tile_objective_value, tile_read_size_w, tile_read_size_h, nr_tiles, use_tiss_mask, tiss_level, tiss_cutoff):
    output_dir = opts_in['output_dir']
    wsi_input = opts_in['wsi_input']

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(wsi_input):
        files_all = sorted(glob.glob(os.path.join(wsi_input, file_name_pattern)))
        with Pool(num_cpu) as p:
            p.map(partial(single_file_run,
                          output_dir=output_dir,
                          input_dir=wsi_input,
                          tile_objective_value=tile_objective_value,
                          tile_read_size_w=tile_read_size_w,
                          tile_read_size_h=tile_read_size_h,
                          nr_tiles=nr_tiles,
                          use_tiss_mask=use_tiss_mask,
                          tiss_level=tiss_level,
                          tiss_cutoff =tiss_cutoff),
                  files_all)

    if os.path.isfile(wsi_input):
        input_dir, file_name = os.path.split(wsi_input)
        single_file_run(file_name=file_name,
                        output_dir=output_dir,
                        input_dir=input_dir,
                        tile_objective_value=tile_objective_value,
                        tile_read_size_w=tile_read_size_w,
                        tile_read_size_h=tile_read_size_h,
                        nr_tiles=nr_tiles,
                        use_tiss_mask=use_tiss_mask,
                        tiss_level=tiss_level,
                        tiss_cutoff=tiss_cutoff)
