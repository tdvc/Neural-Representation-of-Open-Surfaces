# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Open shapes 

Purpose:
Remove the parts of the reconstructed closed shapes that should be holes
"""

# Geometry libraries
from pygel3d import hmesh

# Deep Learning
#from __future__ import print_function, division
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.neural_network import data 

from utils.general import parse_cfg_file

from utils.neural_network import Net   
from utils.geometry import gwn_grad_vector_ssdf, gwn_grad_vector_gwn
from utils.geometry import smooth_boundary_removal

# Math
import numpy as np
array = np.array

# System libraries
import os
from dotenv import load_dotenv

def check_if_triangle_mesh(mesh):
    for f in mesh.faces():
        if (not len(mesh.circulate_face(f,mode='v')) == 3):
            return False
    return True

# ---------------------------------------------------------------------------- #
# Evaluate the reconstructions of the training data set
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

    assert len(os.sys.argv) > 2, \
        "Plese indicate which dataset 'train' or 'test' to use"

    assert (os.sys.argv[2] == "train" or os.sys.argv[2] == "test"), \
        "Plese indicate which dataset 'train' or 'test' to use"

    assert len(os.sys.argv) > 3, \
        "Plese indicate which type either ssdf, gwn, udf"

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
    model_train_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[3] + '_model_train_dir'))
    shape_reconstruction_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[3],os.sys.argv[2],"closed_shapes")

    if (os.sys.argv[2] == "train"):
        mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_train_dir'))
        model_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[3]+'_model_train_dir'))
        model_name = cfg.model_train.name
    else:
        mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_test_dir'))
        model_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[3]+'_model_test_dir'))
        model_name = cfg.model_test.name

    # ---------------------------------------------------------------------------- #
    # If GPU should be used, Check if GPU is available
    # ---------------------------------------------------------------------------- #
    if (cfg.use_GPU):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    # ---------------------------------------------------------------------------- #
    # If GPU should be used, Check if GPU is available
    # ---------------------------------------------------------------------------- #
    checkpoint_latent_vector = torch.load(os.path.join(os.getcwd(),model_dir,model_name),map_location=device)
    model_nr_epoch_latent_vector = checkpoint_latent_vector['epoch']
    model_loss_latent_vector = checkpoint_latent_vector['loss']

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    filelist_meshes = open(os.path.join(mesh_dir,"filelist.txt"), "r")
    mesh_files = obtain_data(filelist_meshes)

    # ---------------------------------------------------------------------------- #
    # Status file
    # ---------------------------------------------------------------------------- #
    open_reconstructed_shapes_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[3],os.sys.argv[2], "open_shapes")
    os.mkdir(open_reconstructed_shapes_dir)
    status_file = os.path.join(open_reconstructed_shapes_dir,"status_file_open_shapes.txt")
    f = open(status_file, "w")
    f.write("Reconstructing the open shapes\n")
    f.write("Using " + str(os.sys.argv[3]) + " reconstruction\n\n")
    f.close()
    f.close()

    # ---------------------------------------------------------------------------- #
    # Load GWN threshold
    # ---------------------------------------------------------------------------- #
    path_gwn_threshold = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[3],"train","closed_shapes","optimal_gwn_threshold.npy")
    assert (os.path.exists(path_gwn_threshold)), \
        "Please find the optimal gwn threshold first"
    gwn_threshold = float(np.load(path_gwn_threshold))

    # ---------------------------------------------------------------------------- #
    # Network
    # ---------------------------------------------------------------------------- #
    if (os.sys.argv[3] == "ssdf"):
        network_output = cfg.network.multiple_output
    else:
        network_output = cfg.network.single_output

    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, network_output, device)
    checkpoint = torch.load(os.path.join(model_train_dir,cfg.model_train.name),map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    net = net.to(device)
    net.normalize_params()
    net.eval()

    shape_analysis_data = data(mesh_dir, model_dir, model_name, device) 

    for ii in range(len(mesh_files)):

        assert(os.path.exists(os.path.join(shape_reconstruction_dir,mesh_files[ii].split(".")[0] + "_closed_reconstructed.obj"))), \
            "File " + os.path.join(shape_reconstruction_dir,mesh_files[ii].split(".")[0] + "_closed_reconstructed.obj") + " does not exist"

        rm = hmesh.obj_load(os.path.join(shape_reconstruction_dir,mesh_files[ii].split(".")[0] + "_closed_reconstructed.obj"))

        _, _, latent_vector = shape_analysis_data[mesh_files[ii]]

        if (os.sys.argv[3] == "ssdf"):
            # ---------------------------------------------------------------------------- #
            # Detect and remove holes
            # ---------------------------------------------------------------------------- #
            gwn_gradient = gwn_grad_vector_ssdf(device, model_train_dir, rm, latent_vector)
            reconstructed_mesh = smooth_boundary_removal(rm, gwn_gradient, gwn_threshold)
        
        elif (os.sys.argv[3] == "gwn"):
            # ---------------------------------------------------------------------------- #
            # Detect and remove holes
            # ---------------------------------------------------------------------------- #
            gwn_gradient = gwn_grad_vector_gwn(device, model_train_dir, rm, latent_vector)
            reconstructed_mesh = smooth_boundary_removal(rm, gwn_gradient, gwn_threshold)

        hmesh.obj_save(os.path.join(open_reconstructed_shapes_dir, mesh_files[ii].split(".")[0] + "_open_reconstructed.obj"),reconstructed_mesh)
        f = open(status_file, "a")
        f.write("Reconstructed mesh " + mesh_files[ii] + " as " + mesh_files[ii].split(".")[0] + "_open_reconstructed.obj" +  "\n")
        f.close()

    f = open(status_file, "a")
    f.write("\n")
    f.write("Done with reconstructing all open shapes\n")
    f.close()