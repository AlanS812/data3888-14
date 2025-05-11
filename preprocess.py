import os

#----------RENAME FILES TO AVOID CONFLICT------------

# def rename_files(base_dir, dir_groups):
#     for group in dir_groups:
#         for directory in group:
#             dir_path = os.path.join(base_dir, directory)
#             if os.path.isdir(dir_path):
#                 for filename in os.listdir(dir_path):
#                     if filename.startswith("cell_") and filename.endswith("_100.png"):
#                         old_path = os.path.join(dir_path, filename)
#                         number = filename.split("_")[1]
#                         new_filename = f"{directory}_{number}.png"
#                         new_path = os.path.join(dir_path, new_filename)
#                         os.rename(old_path, new_path)
                        

# base_dir = "100"

# tumour_dirs = ["Invasive_Tumor", "Prolif_Invasive_Tumor", "T_Cell_and_Tumor_Hybrid"]
# immune_dirs = ["CD4+_T_Cells", "CD8+_T_Cells", "B_Cells", "Mast_Cells", "Macrophages_1", "Macrophages_2", "LAMP3+_DCs", "IRF7+_DCs"]
# stromal_dirs = ["Stromal", "Stromal_and_T_Cell_Hybrid", "Perivascular-Like"]
# other_dirs = ["Endothelial", "Myoepi_ACTA2+", "Myoepi_KRT15+", "DCIS_1", "DCIS_2"]

# all_dir_groups = [tumour_dirs, immune_dirs, stromal_dirs, other_dirs]

# rename_files(base_dir, all_dir_groups)

#------------SAMPLE IMAGES AND SAVE------------

# import random

# base_path = "100"

# def get_files_in_dirs(dir_list):
#     files = []
#     for dir_name in dir_list:
#         dir_path = os.path.join(base_path, dir_name)
#         if os.path.isdir(dir_path):
#             for file_name in os.listdir(dir_path):
#                 full_file_path = os.path.join(dir_path, file_name)
#                 if os.path.isfile(full_file_path):
#                     files.append(os.path.join(dir_name, file_name))
#     return files

# tumour_dirs = [
#     "Invasive_Tumor",
#     "Prolif_Invasive_Tumor",
#     "T_Cell_and_Tumor_Hybrid"
# ]

# immune_dirs = [
#     "CD4+_T_Cells",
#     "CD8+_T_Cells", 
#     "B_Cells", 
#     "Mast_Cells", 
#     "Macrophages_1", 
#     "Macrophages_2", 
#     "LAMP3+_DCs",
#     "IRF7+_DCs"
# ]

# stromal_dirs = [
#     "Stromal", 
#     "Stromal_and_T_Cell_Hybrid", 
#     "Perivascular-Like"
# ]

# other_dirs = [
#     "Endothelial",
#     "Myoepi_ACTA2+", 
#     "Myoepi_KRT15+", 
#     "DCIS_1", 
#     "DCIS_2", 
# ]

# tumour_files = get_files_in_dirs(tumour_dirs)
# immune_files = get_files_in_dirs(immune_dirs)
# stromal_files = get_files_in_dirs(stromal_dirs)
# other_files = get_files_in_dirs(other_dirs)

# random.seed(3888)

# sample_size = 4000

# tumour_sample = random.sample(tumour_files, min(sample_size, len(tumour_files)))
# immune_sample = random.sample(immune_files, min(sample_size, len(immune_files)))
# stromal_sample = random.sample(stromal_files, min(sample_size, len(stromal_files)))
# other_sample = random.sample(other_files, min(sample_size, len(other_files)))

# import shutil

# def copy_files(sample, dest_dir):
#     for file_path in sample:
#         src = os.path.join(base_path, file_path)
#         dst = os.path.join(base_path, dest_dir, os.path.basename(file_path))
#         shutil.copy(src, dst)

# copy_files(immune_sample, 'immune')
# copy_files(tumour_sample, 'tumour')
# copy_files(stromal_sample, 'stromal')
# copy_files(other_sample, 'other')

#---------RESIZE IMAGES---------

from PIL import Image

def resize_images(root_folder, size=(224, 224)):
    subfolders = ['immune', 'other', 'stromal', 'tumour']
    
    for subfolder in subfolders:
        folder_path = os.path.join(root_folder, subfolder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png')):
                file_path = os.path.join(folder_path, filename)
                with Image.open(file_path) as img:
                    resized_img = img.resize(size, Image.LANCZOS)
                    resized_img.save(file_path)

resize_images("original")

# blur

from PIL import Image, ImageFilter
import os
import shutil

def blur_images(root_folder, blur_radius):
    subfolders = ['immune', 'other', 'stromal', 'tumour']
    
    for subfolder in subfolders:
        src_folder = os.path.join(root_folder, subfolder)
        dst_folder = os.path.join(root_folder, f'blur_{blur_radius}', subfolder)
        
        for filename in os.listdir(src_folder):
            if filename.lower().endswith(('.png')):
                src_path = os.path.join(src_folder, filename)
                dst_path = os.path.join(dst_folder, filename)
                
                with Image.open(src_path) as img:
                    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    blurred_img.save(dst_path)

# Example usage
blur_images("original", 3)

# contrast