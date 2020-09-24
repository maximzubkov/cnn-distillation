#!/bin/bash
# Simply installing requirements and pulling checkpoints from lfs

echo "Building started"
echo "Install requirements"
pip install -r requirements.txt
echo "Install lfs"
git lfs install
git lfs pull