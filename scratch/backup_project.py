import os
import shutil

def backup():
    src_dir = "c:/Users/sspl260/.gemini/antigravity/scratch/mini-golf-ai"
    dest_dir = os.path.join(src_dir, "backups/Orange Ball 95% accuracy")
    
    # Ensure destination is empty or clean
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    # Allowed extensions to backup code, config, layouts (excluding heavy video/weights/images)
    allowed_exts = {".py", ".yaml", ".yml", ".json", ".html", ".css", ".js", ".md", ".txt"}
    ignored_dirs = {"backups", "__pycache__", ".git", "venv", "outputs", "assets"}

    print(f"Starting backup from {src_dir} to {dest_dir}...")
    copied_files = 0
    
    for root, dirs, files in os.walk(src_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in allowed_exts:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_dir)
                dest_path = os.path.join(dest_dir, rel_path)
                
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
                copied_files += 1
                
    print(f"Backup completed successfully! Copied {copied_files} source files.")

if __name__ == "__main__":
    backup()
