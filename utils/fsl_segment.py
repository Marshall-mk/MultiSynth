import os
import subprocess
import shutil
import argparse
import fnmatch

INPUT_ROOT = "/home/localkm25/Desktop/PhD/code/MultiSynth/data/ixi_test_t2_cleaned/"              # folder shown in screenshot
OUTPUT_ROOT = "/home/localkm25/Desktop/PhD/code/MultiSynth/data/ixi_test_t2_output_seg/"           # folder to save segmentations

def find_nifti_images(folder, patterns=None):
    """Find all T2 NIfTI files inside subject folder that match patterns."""
    images = []
    for f in os.listdir(folder):
        if f.endswith(".nii") or f.endswith(".nii.gz"):
            # If patterns specified, check if filename matches any pattern
            if patterns:
                if any(fnmatch.fnmatch(f, pattern) for pattern in patterns):
                    images.append(os.path.join(folder, f))
            else:
                images.append(os.path.join(folder, f))
    return images

def run_fast(input_img, out_dir):
    """Run segmentation tool for tissue segmentation."""
    # Get the base name without extension and add _seg suffix
    input_basename = os.path.basename(input_img)
    if input_basename.endswith('.nii.gz'):
        name_without_ext = input_basename[:-7]  # Remove .nii.gz
    elif input_basename.endswith('.nii'):
        name_without_ext = input_basename[:-4]  # Remove .nii
    else:
        name_without_ext = os.path.splitext(input_basename)[0]
    
    base = os.path.join(out_dir, f"{name_without_ext}_seg")
    
    # Set up FreeSurfer environment
    env = os.environ.copy()
    
    # Create a temporary scaled image since input is normalized 0-1
    temp_scaled = os.path.join(out_dir, f"temp_scaled_{name_without_ext}.mgz")
    
    # Try to find mri_synthseg and set FREESURFER_HOME
    mri_synthseg_path = shutil.which("mri_synthseg")
    if mri_synthseg_path:
        freesurfer_home = os.path.dirname(os.path.dirname(mri_synthseg_path))
        env['FREESURFER_HOME'] = freesurfer_home
        
        # First convert and scale the image to FreeSurfer-compatible intensities
        convert_cmd = f"mri_convert {input_img} {temp_scaled} --scale 255"
        print("Scaling image:", convert_cmd)
        try:
            subprocess.run(convert_cmd, shell=True, check=True, env=env)
            
            # Now run segmentation on the scaled image
            cmd = f"mri_synthseg --i {temp_scaled} --o {base}.nii.gz --fast"
            print("Running:", cmd)
            subprocess.run(cmd, shell=True, check=True, env=env)
            
            # Clean up temporary file
            if os.path.exists(temp_scaled):
                os.remove(temp_scaled)
            return
        except subprocess.CalledProcessError as e:
            print(f"FreeSurfer pipeline failed: {e}")
            if os.path.exists(temp_scaled):
                os.remove(temp_scaled)
    
    # Fallback: try with common FreeSurfer paths
    freesurfer_paths = [
        "/usr/local/freesurfer/bin/mri_synthseg",
        "/opt/freesurfer/bin/mri_synthseg", 
        "~/freesurfer/bin/mri_synthseg"
    ]
    
    for fs_path in freesurfer_paths:
        expanded_path = os.path.expanduser(fs_path)
        if os.path.exists(expanded_path):
            # Set FREESURFER_HOME to the parent directory of bin
            freesurfer_home = os.path.dirname(os.path.dirname(expanded_path))
            env['FREESURFER_HOME'] = freesurfer_home
            
            # Try convert path
            convert_path = os.path.join(freesurfer_home, "bin", "mri_convert")
            
            try:
                # First convert and scale the image
                convert_cmd = f"{convert_path} {input_img} {temp_scaled} --scale 255"
                print("Scaling image:", convert_cmd)
                subprocess.run(convert_cmd, shell=True, check=True, env=env)
                
                # Now run segmentation
                cmd = f"{expanded_path} --i {temp_scaled} --o {base}.nii.gz --fast"
                print("Running:", cmd)
                print(f"FREESURFER_HOME set to: {freesurfer_home}")
                subprocess.run(cmd, shell=True, check=True, env=env)
                
                # Clean up temporary file
                if os.path.exists(temp_scaled):
                    os.remove(temp_scaled)
                return
            except subprocess.CalledProcessError as e:
                print(f"Command failed: {e}")
                if os.path.exists(temp_scaled):
                    os.remove(temp_scaled)
                continue
    
    print(f"ERROR: mri_synthseg not found or not working. Please check FreeSurfer installation.")
    raise RuntimeError("No segmentation tool available")

def main():
    parser = argparse.ArgumentParser(description='Run brain segmentation on NIfTI files')
    parser.add_argument('--patterns', nargs='+', 
                        help='File name patterns to process (e.g., "model_*prediction.nii.gz" "eclare_output.nii.gz")')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip files that already have segmentation outputs')
    
    args = parser.parse_args()
    
    for root, dirs, files in os.walk(INPUT_ROOT):
        # Only process leaf folders that contain images
        if any(f.endswith(".nii.gz") for f in files):

            subj_dir = os.path.relpath(root, INPUT_ROOT)
            out_dir = os.path.join(OUTPUT_ROOT, subj_dir)
            os.makedirs(out_dir, exist_ok=True)

            images = find_nifti_images(root, args.patterns)
            if not images:
                if args.patterns:
                    print(f"No images matching patterns {args.patterns} found in {root}")
                else:
                    print(f"No NIfTI images found in {root}")
                continue

            print(f"\n=== Processing {root} ===")
            for img in images:
                img_name = os.path.basename(img)
                
                # Check if output already exists
                if args.skip_existing:
                    if img_name.endswith('.nii.gz'):
                        name_without_ext = img_name[:-7]
                    elif img_name.endswith('.nii'):
                        name_without_ext = img_name[:-4]
                    else:
                        name_without_ext = os.path.splitext(img_name)[0]
                    
                    output_path = os.path.join(out_dir, f"{name_without_ext}_seg.nii.gz")
                    if os.path.exists(output_path):
                        print(f"Skipping {img_name} - output already exists")
                        continue
                
                print(f"Processing: {img_name}")
                run_fast(img, out_dir)

if __name__ == "__main__":
    main()
