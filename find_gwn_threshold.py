# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
GWN threshold

Purpose:
Find a suitable GWN Gradient length threshold
"""

# Geometry libraries
from pygel3d import hmesh

# Deep Learning
#from __future__ import print_function, division
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from utils.general import parse_cfg_file

from utils.neural_network import data
from utils.neural_network import Net   

from utils.geometry import smooth_boundary_removal
from utils.geometry import calculate_chamfer_distance
from utils.geometry import gwn_grad_vector_ssdf, gwn_grad_vector_gwn

# Math
import math
import numpy as np
array = np.array
from bisect import bisect_left

# System libraries
import os
import datetime
from dotenv import load_dotenv
import pandas as pd
from scipy.spatial import KDTree
from random import sample

import shutil

import random
random.seed(42)

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

def chamfer_distance_total(gwn_threshold,*args):

    gtm_KDTree_list = args[0]

    closed_rm_sample_point_path_list = args[1]

    gwn_gradient_vector_list = args[2]

    total_distance_error = 0.0

    status_file = args[4]

    f = open(status_file, "a")
    f.write("Investigating threshold: " + str(gwn_threshold) + "\n")
    f.close()

    for k in range(len(gtm_KDTree_list)):

        gtm_KDTree = gtm_KDTree_list[k]

        gwn_gradient = np.load(gwn_gradient_vector_list[k])

        closed_rm_sample_points = np.load(closed_rm_sample_point_path_list[k])

        sample_points_above_threshold = closed_rm_sample_points[gwn_gradient > gwn_threshold]

        #---------------------------------
        # Distance from open rm to gtm 
        #---------------------------------
        if (len(sample_points_above_threshold) > 0):
            tree_rm = KDTree(sample_points_above_threshold)
            d_rm,_ = tree_rm.query(gtm_KDTree.data, k=1, eps=0, p=2, distance_upper_bound=math.inf)

            d_gtm,_ = gtm_KDTree.query(sample_points_above_threshold, k=1, eps=0, p=2, distance_upper_bound=math.inf)

            distance_error = (np.dot(d_rm,d_rm) + np.dot(d_gtm,d_gtm))/len(sample_points_above_threshold)
        else:
            distance_error = 10000000.0

        total_distance_error += distance_error

        f = open(status_file, "a")
        f.write("Evaluating file " + str(k) + "\n")
        f.write("Using  " + str(len(sample_points_above_threshold)) + " out of " + str(len(closed_rm_sample_points)) + " number of points\n")
        f.write("Error is: " + str(distance_error) + "\n")
        f.close()

    f = open(status_file, "a")
    f.write("Total distance for mesh is: " + str(total_distance_error) + "\n")
    f.close()
    return total_distance_error


# Calculate GWN Threshold
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
        "Plese indicate which type either 'ssdf' or 'gwn'"

    # ---------------------------------------------------------------------------- #
    # Get environment file
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(".env"), \
        "Please create an .env file with appropriate directories and specification of configuration file"
    load_dotenv()
    
    # ---------------------------------------------------------------------------- #
    # Set the directories for loading data, saving model and status file
    # ---------------------------------------------------------------------------- #
    mesh_train_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_train_dir'))
    model_train_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2] + '_model_train_dir'))
    shape_reconstruction_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[2],"train","closed_shapes")

    status_file = os.path.join(shape_reconstruction_dir,"find_gwn_status_file.txt")
    f = open(status_file, "w")
    f.write("Find optimal gwn threshold for experiment " + os.sys.argv[1] + " using method " + os.sys.argv[2] + " for data set train\n")
    f.write("Saving status file as " + status_file + "\n")
    f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    f.write("Using same random seed\n")
    f.write("Loading meshes from: " + mesh_train_dir + "\n")
    f.write("Using trainig model from: " + model_train_dir + "\n")
    f.close()
    shutil.copyfile(os.path.join(os.getcwd(),"find_gwn_threshold.py"), os.path.join(shape_reconstruction_dir,"find_gwn_threshold.py"))
    shutil.copyfile(os.path.join(os.getcwd(),"utils","geometry.py"), os.path.join(shape_reconstruction_dir,"geometry.py"))

    # ---------------------------------------------------------------------------- #
    # Get configuration file
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
    # sample meshes
    # ---------------------------------------------------------------------------- #
    mesh_samples = []

    if (os.path.exists(os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"data","train","filelist.txt"))):
        file_mesh_samples_list = open(os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"data","train","filelist.txt"),"r")
        
        while True:
            next_line = file_mesh_samples_list.readline().rstrip('\n')
            if next_line:
                mesh_samples.append(next_line)
            if not next_line:
                break
        file_mesh_samples_list.close()

        f = open(status_file, "a")
        f.write("Using mesh samples from already generated sample list\n\n")
        f.close()

    assert (len(mesh_samples) > 0), \
        "Either use filelist.txt in data/train folder or create a .txt file with sampled meshes first"
    
    # Ground Truth KDTree
    gtm_KDTree_list = [] 
    
    # reconstructhed meshes list
    closed_rm_list = []

    # gwn_grad_vector_list 
    gwn_gradient_vector_list = []

    closed_rm_sample_point_path_list = []

    gtm_mesh_list = []

    shape_analysis_data = data(mesh_train_dir, model_train_dir, cfg.model_train.name, device) 

    median_gwn_gradient_value = np.zeros(len(mesh_samples))

    # ---------------------------------------------------------------------------- #
    # Find data
    # ---------------------------------------------------------------------------- #
    my_bool_variable = False

    for i in range(len(mesh_samples)):

        f = open(status_file, "a")
        f.write("Getting data for file " + mesh_samples[i] + "\n")
        f.write("\n")
        f.close()
        
        # ----------------------------#
        # Ground Truth mesh
        # ----------------------------#
        if (not os.path.exists(os.path.join(mesh_train_dir,mesh_samples[i]+"_gtm_sample_points.npy"))):
            f = open(status_file, "a")
            f.write("Sampled point on ground truth mesh " + mesh_samples[i] + "\n")
            f.close()

            gtm = hmesh.obj_load(os.path.join(mesh_train_dir,mesh_samples[i] + ".obj"))

            # ----------------------------#
            # Ground Truth mesh samples
            # ----------------------------#
            gtm_triangle_areas = []
            for f in gtm.faces():
                gtm_triangle_areas.append(gtm.area(f))

            gtm_triangle_areas = [tri_area / sum(gtm_triangle_areas) for tri_area in gtm_triangle_areas]
            gtm_triangle_cdf = np.cumsum(gtm_triangle_areas)

            gtm_tri_samples = torch.rand(cfg.surface_reconstruction.chamfer_distance_points).numpy()
            gtm_ran_var = torch.rand(2,cfg.surface_reconstruction.chamfer_distance_points).numpy()

            points_gtm = []
            for k in range(cfg.surface_reconstruction.chamfer_distance_points):
                [xi1,xi2] = gtm_ran_var[:,k]
                sqrt_xi1 = math.sqrt(xi1)
                rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
                rn_select = torch.randperm(3).numpy()
                u = rn[rn_select[0]]
                v = rn[rn_select[1]]
                w = rn[rn_select[2]]

                t_index = bisect_left(gtm_triangle_cdf,  gtm_tri_samples[k])
           
                [v1,v2,v3] = np.array(gtm.circulate_face(t_index,mode='v'))

                points_gtm.append(u*gtm.positions()[v1] + v*gtm.positions()[v2] + w*gtm.positions()[v3])
            points_gtm = np.array(points_gtm)

            f = open(status_file, "a")
            f.write("Number of points: " + str(len(points_gtm)))
            f.write("\n")
            f.close()

            np.save(os.path.join(mesh_train_dir,mesh_samples[i]+"_gtm_sample_points.npy"), points_gtm)

        # ----------------------------#
        # Ground Truth KDTree list
        # ----------------------------#
        points_gtm = np.load(os.path.join(mesh_train_dir,mesh_samples[i]+"_gtm_sample_points.npy"))
        gtm_KDTree = KDTree(points_gtm)
        gtm_KDTree_list.append(gtm_KDTree)

        # ----------------------------#
        # Reconstructed mesh
        # ----------------------------#
        file_name_closed_rm = mesh_samples[i] + "_closed_reconstructed.obj"
        if (not os.path.exists(os.path.join(shape_reconstruction_dir,file_name_closed_rm.split(".")[0]+"_sample_points.npy"))):

            f = open(status_file, "a")
            f.write("Sampled point on reconstructed mesh " + mesh_samples[i] + "\n")
            f.close()
            
            # ----------------------------#
            # Make sure the mesh exists
            # ----------------------------#
            assert (os.path.exists(os.path.join(shape_reconstruction_dir,file_name_closed_rm))), \
                "Run shape_reconstruction.py first"

            closed_rm = hmesh.obj_load(os.path.join(shape_reconstruction_dir,file_name_closed_rm))

            # ----------------------------#
            # Reconstructed mesh samples
            # ----------------------------#
            closed_rm_triangle_areas = []
            for f in closed_rm.faces():
                closed_rm_triangle_areas.append(closed_rm.area(f))

            if (not check_if_triangle_mesh(closed_rm)):
                print("Triangle mesh " + mesh_samples[i] + " is not a triangle mesh")

            closed_rm_triangle_areas = [tri_area / sum(closed_rm_triangle_areas) for tri_area in closed_rm_triangle_areas]
            closed_rm_triangle_cdf = np.cumsum(closed_rm_triangle_areas)

            closed_rm_tri_samples = torch.rand(cfg.surface_reconstruction.chamfer_distance_points).numpy()
            closed_rm_ran_var = torch.rand(2,cfg.surface_reconstruction.chamfer_distance_points).numpy()

            points_closed_rm = []
            for k in range(cfg.surface_reconstruction.chamfer_distance_points):
                [xi1,xi2] = closed_rm_ran_var[:,k]
                sqrt_xi1 = math.sqrt(xi1)
                rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
                rn_select = torch.randperm(3).numpy()
                u = rn[rn_select[0]]
                v = rn[rn_select[1]]
                w = rn[rn_select[2]]

                t_index = bisect_left(closed_rm_triangle_cdf, closed_rm_tri_samples[k])
                #t_index = sample_point(closed_rm_triangle_cdf, 0, len(closed_rm_triangle_cdf), closed_rm_tri_samples[k])

                [v1,v2,v3] = np.array(closed_rm.circulate_face(t_index,mode='v'))

                points_closed_rm.append(u*closed_rm.positions()[v1] + v*closed_rm.positions()[v2] + w*closed_rm.positions()[v3])
            points_closed_rm = np.array(points_closed_rm)
            
            np.save(os.path.join(shape_reconstruction_dir,file_name_closed_rm.split(".")[0]+"_sample_points.npy"), points_closed_rm)

        # ----------------------------#
        # Reconstructed mesh list
        # ----------------------------#
        closed_rm_sample_point_path_list.append(os.path.join(shape_reconstruction_dir,file_name_closed_rm.split(".")[0]+"_sample_points.npy"))
  
        # ----------------------------#
        # Latent vector
        # ----------------------------#
        _, _, latent_vector = shape_analysis_data[mesh_samples[i]+".obj"]

        # ----------------------------#
        # Gradient vector 
        # ----------------------------#
        points_closed_rm = np.load(os.path.join(shape_reconstruction_dir,file_name_closed_rm.split(".")[0]+"_sample_points.npy"))

        if (os.sys.argv[2] == "ssdf"):
            gwn_gradient = gwn_grad_vector_ssdf(device, model_train_dir, points_closed_rm, latent_vector)
        else:
            gwn_gradient = gwn_grad_vector_gwn(device, model_train_dir, points_closed_rm, latent_vector)

        median_gwn_gradient_value[i] = np.median(gwn_gradient)
         
        np.save(os.path.join(shape_reconstruction_dir,file_name_closed_rm.split(".")[0]+"_gwn_gradient.npy"), gwn_gradient)

        # ----------------------------#
        # Gradient vector list
        # ----------------------------#
        gwn_gradient_vector_list.append(os.path.join(shape_reconstruction_dir,file_name_closed_rm.split(".")[0]+"_gwn_gradient.npy"))

        f = open(status_file, "a")
        f.write("Obtained data for file " + mesh_samples[i] + "\n")
        f.close()

    # Starting value
    initial_gwn_threshold_global = np.mean(median_gwn_gradient_value)

    f = open(status_file, "a")
    f.write("Using mean of the median of every gradient vector\n")
    f.write("Initial gwn threshold for basinhopping is: " + str(initial_gwn_threshold_global) + "\n\n")
    f.close()

    minimizer_kwargs = {"args": (gtm_KDTree_list, closed_rm_sample_point_path_list, gwn_gradient_vector_list, cfg.surface_reconstruction.chamfer_distance_points, status_file)}

    # Find global minimum
    output_global = basinhopping(chamfer_distance_total, initial_gwn_threshold_global, 
                    minimizer_kwargs=minimizer_kwargs)


    initial_gwn_threshold_local = output_global.x[0]

    f = open(status_file, "a")
    f.write("Initial gwn threshold for Nelder-Mead optimizaiton is: " + str(initial_gwn_threshold_local) + "\n\n")
    f.close()    
    
    # Find local minimum
    output = minimize(chamfer_distance_total, initial_gwn_threshold_local, 
                    args=(gtm_KDTree_list, closed_rm_sample_point_path_list, gwn_gradient_vector_list, cfg.surface_reconstruction.chamfer_distance_points, status_file),
                    method='Nelder-Mead')

    f = open(status_file, "a")
    f.write("The optimal gwn threshold for experiment " + os.sys.argv[1] + " using " + os.sys.argv[2] + " is: " + str(output.x) + "\n")
    f.close()

    # Save the optimal gwn threshold
    np.save(os.path.join(shape_reconstruction_dir,"optimal_gwn_threshold.npy"), output.x)