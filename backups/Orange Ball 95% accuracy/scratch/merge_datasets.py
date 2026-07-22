import os
import shutil

def main():
    src_dir = "auto_dataset_white"
    dest_dir = "auto_dataset"
    prefix = "white_"
    
    src_images = os.path.join(src_dir, "images")
    src_labels = os.path.join(src_dir, "labels")
    
    dest_images = os.path.join(dest_dir, "images")
    dest_labels = os.path.join(dest_dir, "labels")
    
    if not os.path.exists(src_images) or not os.path.exists(src_labels):
        print("ERROR: Source white-ball dataset directories not found!")
        return
        
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)
    
    # Merge images
    image_count = 0
    for filename in os.listdir(src_images):
        src_path = os.path.join(src_images, filename)
        new_filename = prefix + filename
        dest_path = os.path.join(dest_images, new_filename)
        try:
            shutil.copy2(src_path, dest_path)
            image_count += 1
        except Exception as e:
            print(f"Error copying image {filename}: {e}")
            
    # Merge labels
    label_count = 0
    for filename in os.listdir(src_labels):
        src_path = os.path.join(src_labels, filename)
        new_filename = prefix + filename
        dest_path = os.path.join(dest_labels, new_filename)
        try:
            shutil.copy2(src_path, dest_path)
            label_count += 1
        except Exception as e:
            print(f"Error copying label {filename}: {e}")
            
    print(f"\nSUCCESS: Datasets merged successfully!")
    print(f"Copied {image_count} images to '{dest_images}' with prefix '{prefix}'")
    print(f"Copied {label_count} labels to '{dest_labels}' with prefix '{prefix}'")

if __name__ == "__main__":
    main()
