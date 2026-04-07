from tkinter import filedialog
import tkinter as tk
from pathlib import Path

def debug_file_tree():
    root = tk.Tk()
    root.withdraw()
    
    target_dir = filedialog.askdirectory(title="Select the Parent Directory")
    if not target_dir:
        print("No directory selected. Exiting.")
        return

    base_path = Path(target_dir)
    
    # "Set up a clean table header for the terminal"
    print(f"\n{'Extracted [-3:]':<15} | {'Folder Name':<25} | {'Full File Path'}")
    print("-" * 100)

    for yaml_file in base_path.rglob("metrics.y*ml"):
        folder_name = yaml_file.parent.name
        extracted_val = folder_name[-3:]
        
        # "Flag suspicious values that don't look like your 0.0 to 1.0 floats"
        if "." not in extracted_val:
            flag = " <--- SUSPICIOUS"
        else:
            flag = ""
            
        print(f"{extracted_val:<15} | {folder_name:<25} | {yaml_file}{flag}")

if __name__ == "__main__":
    debug_file_tree()