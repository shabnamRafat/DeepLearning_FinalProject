
import os
import csv


print("Okay")

root_dir = '/home/ubuntu/ShabnamSawda/FinalProject/Data/camera_lidar_semantic'
output_file = 'a2d2_image_mask_pairs.csv'

subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
print(subdirs[1])


# Subfolder names
camera_subfolder = 'camera/cam_front_center'
label_subfolder = 'camera/cam_front_center'

pairs = []

for sequence_folder in sorted(os.listdir(root_dir)):
    seq_path = os.path.join(root_dir, sequence_folder)
    print(f"Processing sequence: {seq_path}")

    camera_path = os.path.join(seq_path, camera_subfolder)

    if not os.path.isdir(camera_path):
        print(f"Directory not found: {camera_path}")
        continue

    # Get all files in the directory
    all_files = sorted(os.listdir(camera_path))

    # Filter image files (assuming they're JPG)
    image_files = [f for f in all_files if f.lower().endswith('.png')]

    for img_file in image_files:
        img_path = os.path.join(camera_path, img_file)

        # Construct the corresponding JSON filename by replacing .jpg extension with .json
        json_file = os.path.splitext(img_file)[0] + '.json'
        json_path = os.path.join(camera_path, json_file)

        # Ensure JSON file exists for this image
        if os.path.exists(json_path):
            pairs.append((img_path, json_path))
        else:
            print(f"Warning: No JSON found for image: {img_path}")

print(f"Total valid image-JSON pairs: {len(pairs)}")

# Save to CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'mask_path'])
    writer.writerows(pairs)


