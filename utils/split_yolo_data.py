<<<<<<< HEAD
'''Purpose: 
- Splits a dataset into train, validation, and test sets for YOLO training.
Key Functionality:
- Reads image paths from a train.txt file.
- Splits data based on specified ratios (e.g., 80% train, 10% val, 10% test).
- Moves or copies images and labels to respective directories.'''

=======
>>>>>>> e5614c173 (final changes)
from pathlib import Path
import random
import os
import shutil

# --- CONFIG ---
base_dir = r"D:\Yak-Identification\repos\Yak_Re-ID\data-finetune"
split_root = os.path.join(base_dir, "2")

image_src = os.path.join(split_root, "images", "train")
label_src = os.path.join(split_root, "labels", "train")
train_file = os.path.join(split_root, "train.txt")
output_train = os.path.join(split_root, "new_train.txt")
output_val = os.path.join(split_root, "val.txt")
output_test = os.path.join(split_root, "test.txt")

# Split ratios
val_split_ratio = 0.1
test_split_ratio = 0.1
train_split_ratio = 1 - val_split_ratio - test_split_ratio

# Random seed
random.seed(42)

def load_and_split_data():
    if not os.path.exists(train_file):
        print(f"❌ Missing {train_file}. Aborting.")
        return None, None, None

    with open(train_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"❌ {train_file} is empty. Aborting.")
        return None, None, None

    random.shuffle(lines)
    total = len(lines)
    val_idx = int(total * val_split_ratio)
    test_idx = int(total * (val_split_ratio + test_split_ratio))

    val_lines = lines[:val_idx]
    test_lines = lines[val_idx:test_idx]
    train_lines = lines[test_idx:]

    return train_lines, val_lines, test_lines

def save_split_files(train_lines, val_lines, test_lines):
    splits = {
        "train": (train_lines, output_train),
        "val": (val_lines, output_val),
        "test": (test_lines, output_test)
    }

    for split_name, (lines, out_path) in splits.items():
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"✅ {split_name.capitalize()} split saved to {out_path} ({len(lines)} lines)")

def move_files(train_lines, val_lines, test_lines, move=False):
    """Move or copy image and label files to their respective split directories."""
    splits = {
        "train": train_lines,
        "val": val_lines,
        "test": test_lines
    }

    for split in splits:
        os.makedirs(os.path.join(split_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(split_root, "labels", split), exist_ok=True)

    for split, lines in splits.items():
        count = 0
        for line in lines:
            filename = os.path.basename(line)
            name_no_ext, _ = os.path.splitext(filename)
            labelname = name_no_ext + ".txt"

            src_img = os.path.join(image_src, filename)
            src_lbl = os.path.join(label_src, labelname)

            dst_img = os.path.join(split_root, "images", split, filename)
            dst_lbl = os.path.join(split_root, "labels", split, labelname)

            # Copy or move image
            if os.path.exists(src_img):
                if move:
                    shutil.move(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
            else:
                print(f"⚠️ Image not found: {src_img}")

            # Copy or move label
            if os.path.exists(src_lbl):
                if move:
                    shutil.move(src_lbl, dst_lbl)
                else:
                    shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"⚠️ Label not found: {src_lbl}")

            count += 1

        print(f"✅ {count} items {'moved' if move else 'copied'} for '{split}' split.")


def main():
    train_lines, val_lines, test_lines = load_and_split_data()
    if any(x is None for x in (train_lines, val_lines, test_lines)):
        print("❌ Splitting failed. Aborting.")
        return

    save_split_files(train_lines, val_lines, test_lines)
    move_files(train_lines, val_lines, test_lines, move=True)
    print("🎯 Done.")

if __name__ == "__main__":
    main()
