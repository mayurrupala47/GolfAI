import os
import sys

def main():
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "auto_dataset"
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    previews_dir = os.path.join(dataset_dir, "previews")
    
    if not os.path.exists(previews_dir):
        print(f"ERROR: Previews directory not found: {previews_dir}")
        return
        
    # Get set of all base names present in the previews folder
    # e.g., "frame_000100.jpg" -> "frame_000100"
    valid_bases = set()
    for filename in os.listdir(previews_dir):
        base, _ = os.path.splitext(filename)
        valid_bases.add(base)
        
    print(f"Source of truth (previews) contains: {len(valid_bases)} valid frames.")
    
    # Clean up images folder
    deleted_images = 0
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            base, ext = os.path.splitext(filename)
            if base not in valid_bases:
                file_path = os.path.join(images_dir, filename)
                try:
                    os.remove(file_path)
                    deleted_images += 1
                except Exception as e:
                    print(f"Error removing image {filename}: {e}")
                    
    # Clean up labels folder
    deleted_labels = 0
    if os.path.exists(labels_dir):
        for filename in os.listdir(labels_dir):
            base, ext = os.path.splitext(filename)
            if base not in valid_bases:
                file_path = os.path.join(labels_dir, filename)
                try:
                    os.remove(file_path)
                    deleted_labels += 1
                except Exception as e:
                    print(f"Error removing label {filename}: {e}")
                    
    print(f"\nSUCCESS: Dataset synced with previews!")
    print(f"Deleted {deleted_images} orphaned images from '{images_dir}'")
    print(f"Deleted {deleted_labels} orphaned labels from '{labels_dir}'")

if __name__ == "__main__":
    main()
