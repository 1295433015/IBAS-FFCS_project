import os
import nibabel as nib
import nibabel.processing
from glob import glob
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool


def preprocess_per_case(data_path):
    files = glob(os.path.join(data_path, '*w_composed_filt.nii.gz'))
    if len(files) == 0:
        print(data_path)
        return

    files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[0]), reverse=True)
    file = files[0]
    if not os.path.exists(file):
        print(data_path)
        return

    input_image = nib.load(file)
    resampled_img = nib.processing.resample_to_output(input_image, [2.232143, 2.232143, 3])
    cropdata = resampled_img.get_fdata()[42:138, 15:111, 110:366]# Modify this parameter when the objective changes.
    crop_nii = nib.Nifti1Image(cropdata, resampled_img.affine, resampled_img.header)
    nib.save(crop_nii, os.path.join(data_path, 'w_composed_filt_preprocessed.nii.gz'))


def unwarp_preprocess_per_case(args):
    return preprocess_per_case(*args)


def batch_preprocess(data_root, processes=8):
    patient_ids = os.listdir(data_root)
    args = [[os.path.join(data_root, patient_id)] for patient_id in patient_ids]
    pool = Pool(processes=processes)
    t_prog = tqdm(pool.imap_unordered(unwarp_preprocess_per_case, args), total=len(args))
    for _ in t_prog:
        pass