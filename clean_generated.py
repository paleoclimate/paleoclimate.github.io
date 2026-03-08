#!/usr/bin/env python3
"""
Remove all generated GeoTIFFs and map outputs (HTMLs, PNGs, PDFs)
from GENERATED_GEOTIFFS, GENERATED_IDW_MAPS, and GENERATED_KNN_IDW_MAPS.
Folders are recreated empty.
"""
import os
import shutil

DIRS = (
    'GENERATED_GEOTIFFS',
    'GENERATED_IDW_MAPS',
    'GENERATED_KNN_IDW_MAPS',
)

def main():
    for d in DIRS:
        if os.path.isdir(d):
            for name in os.listdir(d):
                path = os.path.join(d, name)
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"Removed: {path}")
                else:
                    shutil.rmtree(path)
                    print(f"Removed dir: {path}")
            print(f"Cleaned: {d}/")
        else:
            print(f"(skip, not a dir: {d})")
    print("Done.")

if __name__ == '__main__':
    main()
