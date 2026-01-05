#!/usr/bin/env python3
"""
Rename folders to remove .nii and .nii.gz from their names.
"""
import os
import sys

def rename_folders(directory, dry_run=True):
    """Rename folders to remove .nii extensions from names."""
    renamed_count = 0

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return 0

    # Get all items in directory
    items = os.listdir(directory)

    for item in sorted(items):
        item_path = os.path.join(directory, item)

        # Only process directories
        if not os.path.isdir(item_path):
            continue

        # Check if name contains .nii
        if '.nii' not in item:
            continue

        # Create new name by removing .nii.gz and .nii
        new_name = item.replace('.nii.gz', '').replace('.nii', '')

        # Skip if name unchanged
        if new_name == item:
            continue

        new_path = os.path.join(directory, new_name)

        # Check if target already exists
        if os.path.exists(new_path):
            print(f"SKIP (target exists): {item} -> {new_name}")
            continue

        if dry_run:
            print(f"PREVIEW: {item} -> {new_name}")
        else:
            try:
                os.rename(item_path, new_path)
                print(f"RENAMED: {item} -> {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"ERROR: Failed to rename {item}: {e}")

    return renamed_count

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "test"
    dry_run = "--execute" not in sys.argv

    if dry_run:
        print("=" * 80)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 80)
        print()

    count = rename_folders(directory, dry_run=dry_run)

    print()
    print("=" * 80)
    if dry_run:
        print(f"Found {count} folders that would be renamed")
        print("To actually rename, run: python3 rename_folders.py test --execute")
    else:
        print(f"Successfully renamed {count} folders")
    print("=" * 80)
