# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Shape Analysis 

Purpose:
Analyse the quality of surface reconstructions of the training data and the test data. 
"""

# Geometry libraries
from pygel3d import hmesh

# Deep Learning
#from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader

from utils.general import parse_cfg_file

from utils.geometry import bbox_dims


# Math
import math
import numpy as np
array = np.array

# System libraries
import os
import datetime
from dotenv import load_dotenv
import shutil
import pandas as pd

def check_if_triangle_mesh(mesh):
    for f in mesh.faces():
        if (not len(mesh.circulate_face(f,mode='v')) == 3):
            return False
    return True

def scale_mesh(mesh, scale_factor):
    for v in mesh.vertices():
        mesh.positions()[v] *= scale_factor

def center_mesh(mesh):
    mean_vector = np.mean(mesh.positions(),0)
    for v in mesh.vertices():
        mesh.positions()[v] -= mean_vector

# ---------------------------------------------------------------------------- #
# Find the mesh files
# ---------------------------------------------------------------------------- #
def obtain_data(filelist):
    meshes = []
    while True:
        next_line = filelist.readline().rstrip('\n')
        if next_line:
            meshes.append(next_line + ".obj")
        if not next_line:
            break
    return meshes

if __name__ == '__main__':
    # ---------------------------------------------------------------------------- #
    # Set working directory to be the directory of this file
    # ---------------------------------------------------------------------------- #
    abspath = os.path.abspath(__file__)
    script_directory = os.path.dirname(abspath)
    os.chdir(script_directory)

    # ---------------------------------------------------------------------------- #
    # System arguments
    # ---------------------------------------------------------------------------- #
    assert len(os.sys.argv) > 1, \
        "Plese indicate which experiment to run"

    assert (os.path.exists(os.path.join(os.getcwd(),"experiments",os.sys.argv[1]))), \
        "Plese indicate a valid experiment name"


    # ---------------------------------------------------------------------------- #
    # Get environment, directories and configuration file
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(".env"), \
        "Please create an .env file with appropriate directories and specification of configuration file"
    load_dotenv()

    # ---------------------------------------------------------------------------- #
    # Configuration file
    # ---------------------------------------------------------------------------- #
    cfg_file = os.path.join(os.getcwd(),"experiments", os.sys.argv[1],os.getenv('cfg_file'))
    assert os.path.exists(cfg_file), \
        "Make sure you have the config file (.yaml) in the cfgs folder and set correct path in .env file"
    cfg = parse_cfg_file(cfg_file)

    # ---------------------------------------------------------------------------- #
    # Set the directories for loading data, saving model and status file
    # ---------------------------------------------------------------------------- #
    mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[2])
    gtm_mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"data","test")
    shutil.copyfile(os.path.join(os.getcwd(),"rescale_udf.py"), os.path.join(mesh_dir,"rescale_udf.py"))

    # ---------------------------------------------------------------------------- #
    # Statusfile
    # ---------------------------------------------------------------------------- #
    status_file = os.path.join(mesh_dir,cfg.status_file_shape_reconstruction)
    f = open(status_file, "a")
    f.write("Running rescale_udf script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    f.close()

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    print("Rescaling obj files")
    filelist_meshes = open(os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv("mesh_test_dir"),"filelist.txt"), "r")
    mesh_files = obtain_data(filelist_meshes)

    for i in range(len(mesh_files)):

        file_name = mesh_files[i].split(".")[0]

        print("File_name: ", file_name)

        m = hmesh.obj_load(os.path.join(mesh_dir,file_name + "_binary_unscaled.obj"))
        
        #scale_factor = csv_file.scale_factor[csv_file.file.tolist().index(file_name)]

        #bbox = hmesh.bbox(m)
        #diag = math.sqrt(np.dot(bbox[1]-bbox[0],bbox[1]-bbox[0]))
        #original_dimension = (1.0-1e-2)/scale_factor
        
        # NDC assumes that the unit cube is divided into 128x128x128
        # When reconstructing that they assume each cell is 1 unit,
        # so one has to divide by 1.0/128.0 to get the original scale
        #bbox, dims = bbox_dims(gtm_mesh, 150, "True", epsilon=0.2)

        length = 2.0
        width = 2.0
        height = 2.0
        bbox = np.array([[-1.0, -1.0, -1.0],
                        [1.0, 1.0, 1.0]])

        offset = (bbox[1]+bbox[0])/2 + np.array([0.5,0.5,0.5])
        #offset = np.array([0.5,0.5,0.5])

        # Scale
        scale_mesh(m,1.0/(150.0))

        # Center
        for v in m.vertices():
            m.positions()[v] = m.positions()[v] - np.array([0.5,0.5,0.5]) + (bbox[1]+bbox[0])/2

        L = max(length, max(width, height))

        scale_mesh(m,L)

        f = open(status_file, "a")
        f.write("Scaling mesh " + file_name + " with scaling constant 150\n")
        f.write("L scaling is: " + str(L) + " \n")
        f.write("Bounding box is: " + str(bbox) + "\n")
        f.write("\n")
        f.close()

        hmesh.obj_save(os.path.join(mesh_dir,file_name + "_reconstructed.obj"),m)