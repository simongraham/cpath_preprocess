import os
from preprocessing.tissuemask import tiss_mask_multi_cpu


def gen_tissue_mask(wsi_input,
        output_dir='tissueMask',
        file_name_pattern='*.ndpi',
        num_cpu=4,
        tiss_level=5
        ):

    if output_dir is None:
        if os.path.isfile(wsi_input):
            dir_path, _ = os.path.split(wsi_input)
            output_dir = os.path.join(dir_path, '..', 'tissueMask')
        elif os.path.isdir(wsi_input):
            output_dir = os.path.join(wsi_input, '..', 'tissueMask')

    opts = {
        'output_dir': output_dir,
        'wsi_input': wsi_input,
    }

    tiss_mask_multi_cpu.run(opts_in=opts,
                             file_name_pattern=file_name_pattern,
                             num_cpu=num_cpu,
                             tiss_level=tiss_level)
