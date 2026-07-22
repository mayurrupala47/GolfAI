import os
import shutil
import random

def prepare_dataset():
    base_dir = "c:/Users/sspl260/.gemini/antigravity/scratch/mini-golf-ai"
    auto_images_dir = os.path.join(base_dir, "auto_dataset/images")
    auto_labels_dir = os.path.join(base_dir, "auto_dataset/labels")
    
    # Destination directory structure
    dataset_dest = os.path.join(base_dir, "datasets/golf_dataset")
    train_img_dest = os.path.join(dataset_dest, "train/images")
    train_lbl_dest = os.path.join(dataset_dest, "train/labels")
    val_img_dest = os.path.join(dataset_dest, "val/images")
    val_lbl_dest = os.path.join(dataset_dest, "val/labels")
    
    # Clean and create destination folders
    if os.path.exists(dataset_dest):
        shutil.rmtree(dataset_dest)
    os.makedirs(train_img_dest, exist_ok=True)
    os.makedirs(train_lbl_dest, exist_ok=True)
    os.makedirs(val_img_dest, exist_ok=True)
    os.makedirs(val_lbl_dest, exist_ok=True)
    
    # Scan for matching pairs
    image_files = [f for f in os.listdir(auto_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    valid_pairs = []
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        lbl_file = f"{base_name}.txt"
        lbl_path = os.path.join(auto_labels_dir, lbl_file)
        
        if os.path.exists(lbl_path):
            valid_pairs.append((img_file, lbl_file))
            
    print(f"Found {len(valid_pairs)} valid image-label pairs.")
    
    if len(valid_pairs) == 0:
        print("Error: No matching image-label pairs found!")
        return
        
    # Shuffle and split (85% Train, 15% Val)
    random.seed(42)
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * 0.85)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"Splitting: {len(train_pairs)} train items, {len(val_pairs)} validation items...")
    
    # Copy train split
    for img_f, lbl_f in train_pairs:
        shutil.copy2(os.path.join(auto_images_dir, img_f), os.path.join(train_img_dest, img_f))
        shutil.copy2(os.path.join(auto_labels_dir, lbl_f), os.path.join(train_lbl_dest, lbl_f))
        
    # Copy val split
    for img_f, lbl_f in val_pairs:
        shutil.copy2(os.path.join(auto_images_dir, img_f), os.path.join(val_img_dest, img_f))
        shutil.copy2(os.path.join(auto_labels_dir, lbl_f), os.path.join(val_lbl_dest, lbl_f))
        
    # Create data.yaml
    yaml_content = """# Custom Golf Ball Dataset Config
path: /content/datasets/golf_dataset  # Absolute path on Google Colab
train: train/images
val: val/images

nc: 1
names: ['golf-ball']
"""
    with open(os.path.join(dataset_dest, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print("Created data.yaml.")
    
    # Zip the resulting folder
    zip_output = os.path.join(base_dir, "datasets/golf_dataset")
    shutil.make_archive(zip_output, 'zip', dataset_dest)
    print(f"Compressed dataset ready: {zip_output}.zip")

if __name__ == "__main__":
    prepare_dataset()
