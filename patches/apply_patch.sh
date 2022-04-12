#!/bin/bash
SCRIPT_PATH=`realpath "$0"`
SCRIPT_DIR=`dirname "$SCRIPT_PATH"`
yes | cp -rf $SCRIPT_DIR/torch_geometric_loader_utils.py ~/miniconda3/envs/fashion/lib/python3.9/site-packages/torch_geometric/loader/utils.py