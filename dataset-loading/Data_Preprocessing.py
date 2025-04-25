import os
import csv
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize

#
# 1) CSV GENERATOR: look for _mask.png instead of .json
#
root_dir    = '../Data/camera_lidar_semantic'
output_file = 'a2d2_image_mask_pairs.csv'
camera_sub  = 'camera/cam_front_center'

pairs = []
for seq in sorted(os.listdir(root_dir)):
    seq_path    = os.path.join(root_dir, seq)
    cam_path    = os.path.join(seq_path, camera_sub)
    if not os.path.isdir(cam_path):
        continue

    for fn in sorted(os.listdir(cam_path)):
        if not fn.lower().endswith('.png'):
            continue
        img_path  = os.path.join(cam_path, fn)
        mask_fn   = os.path.splitext(fn)[0] + '_mask.png'
        mask_path = os.path.join(cam_path, mask_fn)

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            # Fall back to JSON if mask PNG missing
            json_fn   = os.path.splitext(fn)[0] + '.json'
            json_path = os.path.join(cam_path, json_fn)
            if os.path.exists(json_path):
                pairs.append((img_path, json_path))
            else:
                print(f"⚠️  no mask for {img_path}")

with open(output_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['image_path','mask_path'])
    w.writerows(pairs)


#
# 2) DATASET CLASS: read PNG masks—or rasterize JSON polygons
#
class A2D2_CSV_dataset(VisionDataset):
    def __init__(self, csv_file, class_list, cache,
                 height, width, transform, target_transform):
        super().__init__(None, transform=transform, target_transform=target_transform)

        # build color→id map
        with open(class_list) as f:
            cls = json.load(f)

        hex2rgb = {
            hex_code: tuple(
                int(hex_code.strip("#")[i: i + 2], 16)
                for i in (0, 2, 4)
            )
            for hex_code in cls.keys()
        }

        # invert
        self.rgb2ids = {rgb:i for i,rgb in enumerate(hex2rgb.values())}

        self.height, self.width, self.cache = height, width, cache

        self.record_list = []
        with open(csv_file) as f:
            r = csv.DictReader(f)
            for row in r:
                self.record_list.append((row['image_path'], row['mask_path']))

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        img_path, label_path = self.record_list[idx]

        # 1) load & transform image
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = self.transform(image)

        # 2) build mask
        if label_path.lower().endswith('.png'):
            # straightforward: read the colored mask PNG
            label = read_image(label_path, mode=ImageReadMode.RGB)
            label = self.target_transform(label)
            mask = torch.zeros(self.height, self.width, dtype=torch.int64)
            for rgb, cid in self.rgb2ids.items():
                m = (label == torch.tensor(rgb).view(3,1,1)).all(dim=0)
                mask[m] = cid

        else:
            # rasterize JSON annotations
            with open(label_path) as f:
                ann = json.load(f)

            # empty int mask
            mask_np = np.zeros((self.height, self.width), dtype=np.int64)
            for shape in ann.get('shapes', []):
                # adapt these keys to your JSON schema!
                # assume shape['label'] is a HEX color, shape['points'] is [[x,y],...]
                rgb = tuple(int(shape['label'].strip('#')[i:i+2],16) for i in (0,2,4))
                cid = self.rgb2ids[rgb]
                poly = shape['points']

                # draw polygon
                img_pil = Image.new('L', (self.width, self.height), 0)
                ImageDraw.Draw(img_pil).polygon(poly, outline=cid, fill=cid)
                mask_np |= np.array(img_pil)

            mask = torch.from_numpy(mask_np)

        # normalize image & return
        return image.float().div(255), mask

#
# 3) EXAMPLE USAGE (make sure to pass these args in your training script)
#
if __name__ == "__main__":
    from torchvision.transforms import Compose

    # example paths — replace with your argparse args
    csv_file         = 'a2d2_image_mask_pairs.csv'
    class_list       = 'class_list.json'
    cache            = '/tmp/cache'
    height, width    = 1208, 1920

    # simple resize transforms
    image_transform  = Resize((height, width), interpolation=InterpolationMode.BILINEAR)
    target_transform = Resize((height, width), interpolation=InterpolationMode.NEAREST)

    ds = A2D2_CSV_dataset(
        csv_file, class_list, cache,
        height, width, image_transform, target_transform
    )

    print("Found", len(ds), "samples")
    im, m = ds[0]
    print("Image:", im.shape, "Mask:", m.shape)
