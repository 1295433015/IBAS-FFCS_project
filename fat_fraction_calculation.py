#对标签进行腐蚀,逐像素计算FF值
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.morphology import ball, erosion, disk
from glob import glob

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def read_image(file_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def calculate_fat_fraction(fat_array, water_array, mask_array, selem):
    unique_labels = np.unique(mask_array)
    fat_fractions = []
    
    for label in unique_labels:
        if label == 0 or label > 10:
            continue
        
        mask_label = (mask_array == label)
        
        # Erode the mask
        eroded_mask = erosion(mask_label, selem)
        
        # Calculate fat fraction per pixel
        fat_fraction_per_pixel = np.zeros_like(fat_array, dtype=np.float32)
        valid_pixels = (fat_array + water_array) > 0
        fat_fraction_per_pixel[valid_pixels] = fat_array[valid_pixels] / (fat_array[valid_pixels] + water_array[valid_pixels])
        
        # Use eroded mask to select regions in fat_fraction_per_pixel
        fat_fraction_mean = np.mean(fat_fraction_per_pixel[eroded_mask])
        
        fat_fractions.append({'Label': label, 'Fat_Fraction': fat_fraction_mean})
    
    return pd.DataFrame(fat_fractions)

def process_patient(fat_file, water_file, mask_file, patient_id, selem):
    print(f'Processing patient: {patient_id}')
    try:
        # Read images
        fat_image = sitk.ReadImage(fat_file)
        water_image = sitk.ReadImage(water_file)
        mask_img = sitk.ReadImage(mask_file)
    except RuntimeError as e:
        print(f"Error reading images for patient {patient_id}: {e}")
        return None  # Skip processing for this patient
    # Read images
    fat_image = sitk.ReadImage(fat_file)
    water_image = sitk.ReadImage(water_file)
    mask_img = sitk.ReadImage(mask_file)
 
    if fat_image.GetSize() != water_image.GetSize() or fat_image.GetSize() != mask_img.GetSize():
        print(f'Skipping {patient_id} due to image size mismatch')
        return None
    
    mask_img.CopyInformation(fat_image)
    
    # Convert images to arrays
    fat_array = sitk.GetArrayFromImage(fat_image)
    water_array = sitk.GetArrayFromImage(water_image)
    mask_array = sitk.GetArrayFromImage(mask_img).astype(np.uint8)  # Convert mask array to uint8

    return calculate_fat_fraction(fat_array, water_array, mask_array, selem)

def batch_calculate(fat_root, water_root, mask_root, save_file):
    ensure_directory_exists(os.path.dirname(save_file))
    
    results = []
    
    fat_files = sorted(glob(os.path.join(fat_root, '*')))
    water_files = sorted(glob(os.path.join(water_root, '*')))
    mask_files = sorted(glob(os.path.join(mask_root, '*')))
    
    min_length = min(len(fat_files), len(water_files), len(mask_files))
    
    fat_files = fat_files[:min_length]
    water_files = water_files[:min_length]
    mask_files = mask_files[:min_length]
    
    selem = ball(1) if len(fat_files[0].split('.')) == 3 else disk(1)
    
    for fat_file, water_file, mask_file in tqdm(zip(fat_files, water_files, mask_files), total=min_length):
        patient_id = os.path.basename(fat_file)
        # output_dir = os.path.join(os.path.dirname(save_file), f"{patient_id}_eroded_images")
        result = process_patient(fat_file, water_file, mask_file, patient_id, selem)
        
        if result is not None and not result.empty:
            result['Patient_ID'] = patient_id
            results.append(result)
    
    if results:
        result_df = pd.concat(results, ignore_index=True)
        result_df.to_excel(save_file, index=False)
        print(f'Results saved to {save_file}')
    else:
        print("No valid results were generated.")

def run_batch_jobs(jobs):
    for fat_root, water_root, mask_root, save_file in jobs:
        print(f"Processing {save_file} ...")
        batch_calculate(fat_root, water_root, mask_root, save_file)

if __name__ == '__main__':
    jobs = ['path for fat image','path for water image',
            'path for groundtruth','savepath']


    run_batch_jobs(jobs)
