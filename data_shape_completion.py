# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Shape Completition Data Generation

Purpose:
The purpose of this function is to create shapes that look like they
have only been scanned from one side.
"""


import os
from pygel3d import hmesh
from dotenv import load_dotenv
import numpy as np
import datetime

from utils.general import parse_cfg_file

if __name__ == '__main__':
    print("Running shape completition data generation")
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
    # Get environment file
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(".env"), \
        "Please create an .env file with appropriate directories"
    load_dotenv()

    # ---------------------------------------------------------------------------- #
    # mesh directories
    # ---------------------------------------------------------------------------- #
    mesh_test_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_test_dir'))
    mesh_shape_completion_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_shape_completion_dir'))

    save_status_file = os.path.join(mesh_shape_completion_dir,"status_file.txt")
    status_file_f = open(save_status_file, "w")
    status_file_f.write("Status file for shape completion data for " + os.sys.argv[1] + "\n")
    status_file_f.write("Saving status file in " + save_status_file + "\n")
    status_file_f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    status_file_f.close()

    assert os.path.exists(os.path.join(os.getcwd(),mesh_test_dir,"filelist.txt")), \
        "Please make sure there is data in test folder"

    # ---------------------------------------------------------------------------- #
    # configuration file
    # ---------------------------------------------------------------------------- #
    cfg_file = os.path.join(os.getcwd(),"experiments", os.sys.argv[1],os.getenv('cfg_file'))
    assert os.path.exists(cfg_file), \
        "Make sure you have the config file (.yaml) in the cfgs folder and set correct path in .env file"
    cfg = parse_cfg_file(cfg_file)

    meshes = []
    filenames = []
    filelist = open(os.path.join(mesh_test_dir,"filelist.txt"), "r")
    while True:
        next_line = filelist.readline().rstrip('\n')
        if next_line:
            filenames.append(next_line)
            m = hmesh.obj_load(os.path.join(mesh_test_dir,next_line + ".obj"))
            meshes.append(m)
        if not next_line:
            break 
    
    filelist = open(os.path.join(mesh_shape_completion_dir,"filelist.txt"), "w")
    filelist.close()
    for i in range(len(meshes)):
        m = meshes[i]

        status_file_f = open(save_status_file,"a")
        status_file_f.write("Removing one side scan for mesh " + filenames[i] + "\n")
        status_file_f.close()
        q = np.random.randint(low=0,high=m.no_allocated_vertices(), size=1)
        n = m.vertex_normal(q[0])
        for v in m.vertices():
            if (np.dot(m.vertex_normal(v),n) <= 0):
                m.remove_vertex(v)
        hmesh.stitch(m)
        m.cleanup()

        hmesh.obj_save(os.path.join(mesh_shape_completion_dir,filenames[i] + ".obj"),m)
        filelist = open(os.path.join(mesh_shape_completion_dir,"filelist.txt"), "a")
        filelist.write(filenames[i])
        filelist.write("\n")
        filelist.close()
    
    status_file_f = open(save_status_file,"a")
    status_file_f.write("Done generating shape completion data\n")
    status_file_f.close()
