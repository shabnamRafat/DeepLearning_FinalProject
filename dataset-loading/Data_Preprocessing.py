#%%

import os
import csv


root_dir = '/home/ubuntu/ShabnamSawda/FinalProject/Data/camera_lidar_semantic'
output_file = 'a2d2_image_mask_pairs.csv'

subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
print(subdirs)

exit()



# Subfolder names
camera_subfolder = 'camera/cam_front_center'
label_subfolder = 'camera/cam_front_center'

pairs = []

for sequence_folder in sorted(os.listdir(root_dir)):
    seq_path = os.path.join(root_dir, sequence_folder)
    camera_path = os.path.join(seq_path, camera_subfolder)
    label_path = os.path.join(seq_path, label_subfolder)

    if not os.path.isdir(camera_path) or not os.path.isdir(label_path):
        print("Here..")
        continue

    image_filenames = sorted(os.listdir(camera_path))
    # print(image_filenames)

    for img_file in image_filenames:
        img_path = os.path.join(camera_path, img_file)
        lbl_path = os.path.join(label_path, img_file)

        # Ensure mask exists for this image
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))

print(f"Total valid image-mask pairs: {len(pairs)}")

# Save to CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'mask_path'])
    writer.writerows(pairs)



#%%

from PIL import Image

for i, (img_path, lbl_path) in enumerate(pairs[:5]):
    try:
        img = Image.open(img_path)
        mask = Image.open(lbl_path)
    except Exception as e:
        print(f"Error in pair {i}: {e}")
