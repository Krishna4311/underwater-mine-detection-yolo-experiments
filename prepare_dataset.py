from pathlib import Path
import shutil
import random

random.seed(42)

# ---- Stage 1: merge all year-folders into one flat structure ----
source_root = Path('.')          # run from inside the folder containing 2010/, 2015/, etc.
output_root = Path('merged_dataset')
output_images = output_root / 'images'
output_labels = output_root / 'labels'
output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)

MINE_CLASS_INDEX = '0'   # MILCO, confirmed from obj.names.txt

copied = 0
skipped_no_pair = 0

for txt_file in source_root.rglob('*.txt'):
    if txt_file.name in ('obj.names.txt', 'yolov4-custom.txt', 'fold_struct.txt'):
        continue

    img_file = txt_file.with_suffix('.jpg')
    if not img_file.exists():
        skipped_no_pair += 1
        continue

    if txt_file.stat().st_size == 0:
        kept_lines = []
    else:
        kept_lines = [
            line for line in txt_file.read_text().strip().splitlines()
            if line.split()[0] == MINE_CLASS_INDEX
        ]

    dest_txt = output_labels / txt_file.name
    dest_txt.write_text('\n'.join(kept_lines))

    dest_img = output_images / img_file.name
    shutil.copy2(img_file, dest_img)
    copied += 1

print(f"Merged {copied} image/label pairs. Skipped {skipped_no_pair} (no matching image).")

# ---- Stage 2: stratified train/val split ----
VAL_RATIO = 0.2

mine_files = []
background_files = []

for label_file in output_labels.glob('*.txt'):
    if label_file.stat().st_size > 0:
        mine_files.append(label_file.stem)      # non-empty -> contains a mine box
    else:
        background_files.append(label_file.stem)  # empty -> background only

random.shuffle(mine_files)
random.shuffle(background_files)

def split(file_list, val_ratio):
    val_count = int(len(file_list) * val_ratio)
    return file_list[val_count:], file_list[:val_count]   # train, val

mine_train, mine_val = split(mine_files, VAL_RATIO)
bg_train, bg_val = split(background_files, VAL_RATIO)

for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    (output_root / sub).mkdir(parents=True, exist_ok=True)

def move_pair(stem, split_name):
    img = output_images / f'{stem}.jpg'
    lbl = output_labels / f'{stem}.txt'
    shutil.move(str(img), f'{output_root}/images/{split_name}/{img.name}')
    shutil.move(str(lbl), f'{output_root}/labels/{split_name}/{lbl.name}')

for stem in mine_train + bg_train:
    move_pair(stem, 'train')
for stem in mine_val + bg_val:
    move_pair(stem, 'val')

print(f"Mine files   -> train: {len(mine_train)}, val: {len(mine_val)}")
print(f"Background   -> train: {len(bg_train)}, val: {len(bg_val)}")
