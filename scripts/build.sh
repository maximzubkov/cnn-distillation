#!/bin/bash

echo "Building started"
mkdir build
git clone https://github.com/dfdazac/wassdistance.git build
echo "Install requirements"
pip install -r requirements.txt
echo "Install lfs"
git lfs install
git lfs pull