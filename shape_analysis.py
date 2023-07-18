# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Shape Analysis 

Purpose:
Analyse the quality of surface reconstructions of the training data and the test data. 
The quality is assesed using the two metrics: Chamfer Distance and Hausdorff Distance
"""

# Geometry libraries
from pygel3d import hmesh

# Deep Learning
#from __future__ import print_function, division
import torch

from utils.general import parse_cfg_file

from utils.neural_network import data 

from utils.geometry import gwn_grad_vector_ssdf, gwn_grad_vector_gwn
from utils.geometry import smooth_boundary_removal
from utils.geometry import calculate_chamfer_distance_and_mesh_accuracy
from utils.geometry import calculate_mesh_completion


# Math
import numpy as np
array = np.array

# System libraries
import os
import datetime
from dotenv import load_dotenv
import shutil

def check_if_triangle_mesh(mesh):
    for f in mesh.faces():
        if (not len(mesh.circulate_face(f,mode='v')) == 3):
            return False
    return True

def scale_mesh(mesh, scale_factor):
    for v in mesh.vertices():
        mesh.positions()[v] *= scale_factor

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
        "Plese indicate which type either ssdf, gwn, udf or delta"
    
    # ---------------------------------------------------------------------------- #
    # Get environment, directories and configuration file
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(".env"), \
        "Please create an .env file with appropriate directories and specification of configuration file"
    load_dotenv()
    
    # ---------------------------------------------------------------------------- #
    # Set the directories for loading data, saving model and status file
    # ---------------------------------------------------------------------------- #
    mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_test_dir'))
    shape_analysis_sampled_gtm_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_analysis")

    model_test_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2]+'_model_test_dir'))
    model_train_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2] + '_model_train_dir'))

    shape_analysis_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_analysis",os.sys.argv[2])
    shape_reconstruction_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[2],"test","open_shapes")

    # ---------------------------------------------------------------------------- #
    # Configuration file
    # ---------------------------------------------------------------------------- #
    cfg_file = os.path.join(os.getcwd(),"experiments", os.sys.argv[1],os.getenv('cfg_file'))
    assert os.path.exists(cfg_file), \
        "Make sure you have the config file (.yaml) in the cfgs folder and set correct path in .env file"
    cfg = parse_cfg_file(cfg_file)
    

    # ---------------------------------------------------------------------------- #
    # If GPU should be used, Check if GPU is available
    # ---------------------------------------------------------------------------- #
    if (cfg.use_GPU):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    mesh_filelist = open(os.path.join(mesh_dir,"filelist.txt"), "r")
    mesh_list = obtain_data(mesh_filelist)

    shape_analysis_data = data(mesh_dir, model_test_dir, cfg.model_test.name, device) 
  
    # ---------------------------------------------------------------------------- #
    # Status file
    # ---------------------------------------------------------------------------- #
    status_file = os.path.join(shape_analysis_dir,cfg.status_file_shape_analysis)

    # ---------------------------------------------------------------------------- #
    # Evaluate the reconstructions of the training data set
    # ---------------------------------------------------------------------------- #
    chamfer_distance = np.zeros([len(mesh_list)])
    mesh_accuracy = np.zeros([len(mesh_list)])
    mesh_completion = np.zeros([len(mesh_list)])
    hausdorff_distance = np.zeros([len(mesh_list)])

    # ---------------------------------------------------------------------------- #
    # Load GWN threshold
    # ---------------------------------------------------------------------------- #
    path_gwn_threshold = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[2],"train","optimal_gwn_threshold.npy")
    assert (os.path.exists(path_gwn_threshold) or os.sys.argv[2] == "udf"), \
        "Please find the optimal gwn threshold first"
    if (os.sys.argv[2] != "udf"):
        gwn_threshold = float(np.load(path_gwn_threshold))

    # ---------------------------------------------------------------------------- #
    # Write parameters to status file
    # ---------------------------------------------------------------------------- #
    f = open(status_file, "w")
    f.write("Shape analysis for experiment " + os.sys.argv[1] + " using method " + os.sys.argv[2] + " for data set test\n")
    f.write("Saving status file as " + status_file + "\n")
    f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    f.write("Using parameters:\n")
    f.write("Chamfer Distance Points: " + str(cfg.surface_reconstruction.chamfer_distance_points) + "\n")
    f.write("Mesh Accuracy points: " + str(cfg.surface_reconstruction.mesh_accuracy_points) + "\n")
    f.write("Mesh Completion Points: " + str(cfg.surface_reconstruction.mesh_completion_points) + "\n")
    f.write("Delta Distance for mesh completion: " + str(cfg.surface_reconstruction.delta_distance) + "\n")
    if (os.sys.argv[2] != "udf"):
        f.write("Using gwn threshold for hole detection: " + str(gwn_threshold) + "\n")
    f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    # Copy shape analysis script and configuration file
    shutil.copyfile(os.path.join(os.getcwd(),"shape_analysis.py"), os.path.join(shape_analysis_dir,"shape_analysis.py"))
    shutil.copyfile(cfg_file, os.path.join(shape_analysis_dir,"configuration_file.yaml"))

    for i in range(len(mesh_list)):

        assert(os.path.exists(os.path.join(shape_reconstruction_dir,mesh_list[i].split(".")[0] + "_open_reconstructed.obj"))), \
            "File " + os.path.join(shape_reconstruction_dir,mesh_list[i].split(".")[0] + "_open_reconstructed.obj") + " does not exist"

        rm = hmesh.obj_load(os.path.join(shape_reconstruction_dir,mesh_list[i].split(".")[0] + "_open_reconstructed.obj"))
        ground_truth_mesh = hmesh.obj_load(os.path.join(mesh_dir,mesh_list[i]))

        _, _, latent_vector = shape_analysis_data[mesh_list[i]]

        if (os.sys.argv[2] == "ssdf"):
            f = open(status_file, "a")
            f.write("Using ssdf reconstruction\n")
            f.close()

            # ---------------------------------------------------------------------------- #
            # Detect and remove holes
            # ---------------------------------------------------------------------------- #
            gwn_gradient = gwn_grad_vector_ssdf(device, model_train_dir, rm, latent_vector)
            reconstructed_mesh = smooth_boundary_removal(rm, gwn_gradient, gwn_threshold)
        
        elif (os.sys.argv[2] == "gwn"):
            f = open(status_file, "a")
            f.write("Using gwn reconstruction\n")
            f.close()
            
            # ---------------------------------------------------------------------------- #
            # Detect and remove holes
            # ---------------------------------------------------------------------------- #
            gwn_gradient = gwn_grad_vector_gwn(device, model_train_dir, rm, latent_vector)
            reconstructed_mesh = smooth_boundary_removal(rm, gwn_gradient, gwn_threshold)

        elif (os.sys.argv[2] == "udf"):
            f = open(status_file, "a")
            f.write("Using udf reconstruction\n")
            f.close()
            reconstructed_mesh = hmesh.Manifold(rm)
    
        [chamfer_distance[i],mesh_accuracy[i]] = calculate_chamfer_distance_and_mesh_accuracy(ground_truth_mesh, 
                                                                            reconstructed_mesh, cfg.surface_reconstruction.chamfer_distance_points,
                                                                            cfg.surface_reconstruction.mesh_accuracy_points,
                                                                            shape_analysis_sampled_gtm_dir, mesh_list[i])

        mesh_completion[i] = calculate_mesh_completion(ground_truth_mesh,reconstructed_mesh,
                                                                        cfg.surface_reconstruction.mesh_completion_points,
                                                                        cfg.surface_reconstruction.delta_distance,
                                                                        shape_analysis_sampled_gtm_dir, mesh_list[i])
        f = open(status_file, "a")
        f.write("Shape measures for reconstructed file of " + mesh_list[i] + " with index: " + str(i) + "\n")
        f.write("Chamfer distance is: " + str(chamfer_distance[i]) + "\n")
        f.write("Mesh accuracy is: " + str(mesh_accuracy[i]) + "\n")
        f.write("Mesh completion is: " + str(mesh_completion[i]) + "\n\n")
        f.close()

    f = open(status_file, "a")
    f.write("Mean chamfer distance: " + str(np.mean(chamfer_distance)) + "\n")
    f.write("Median chamfer distance: " + str(np.median(chamfer_distance)) + "\n")
    f.write("Mean mesh accuracy: " + str(np.mean(mesh_accuracy)) + "\n")
    f.write("Median mesh accuracy: " + str(np.median(mesh_accuracy)) + "\n")
    f.write("Mean mesh completion: " + str(np.mean(mesh_completion)) + "\n")
    f.write("Median mesh completion: " + str(np.median(mesh_completion)) + "\n")

    np.save(os.path.join(shape_analysis_dir,os.sys.argv[2] + "_chamfer_distance_measurements.npy"), chamfer_distance)
    np.save(os.path.join(shape_analysis_dir,os.sys.argv[2] + "_mesh_accuracy_measurements.npy"), mesh_accuracy)
    np.save(os.path.join(shape_analysis_dir,os.sys.argv[2] + "_mesh_completion_measurements.npy"), mesh_completion)
    np.save(os.path.join(shape_analysis_dir,os.sys.argv[2] + "_hausdorff_distance_measurements.npy"), hausdorff_distance)
    f.write("\n")
    f.close()