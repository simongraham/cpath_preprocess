from preprocessing.tissuemask import tissuegenerator
import os
import glob
from multiprocessing import Pool
from functools import partial


def tissue_all_process(obj):
    obj.load_wsi()
    obj.stain_entropy_otsu()
    obj.morphology()
    obj.save_file()

def single_file_run(file_name, output_dir, input_dir, tiss_level):
    print('Generating tissue mask for ' + os.path.basename(file_name), flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type == '.svs' or file_type == '.ndpi' or file_type == '.mrxs' or file_type == 'tif' or file_type == 'tiff':
        tiss_obj = tissuegenerator.TissueGenerator(input_dir=input_dir,
                                               file_name=file_name,
                                               output_dir=output_dir,
                                               tiss_level=tiss_level)
        tissue_all_process(obj=tiss_obj)


def run(opts_in,
        file_name_pattern, num_cpu, tiss_level):
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
                          tiss_level=tiss_level),
                  files_all)

    if os.path.isfile(wsi_input):
        input_dir, file_name = os.path.split(wsi_input)
        single_file_run(file_name=file_name,
                        output_dir=output_dir,
                        input_dir=input_dir,
                        tiss_level=tiss_level)
