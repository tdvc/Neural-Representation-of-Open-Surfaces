# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Interpolation between two shapes

Purpose:
The purpose of this function is to interpolate between two different shapes
in latent space using the shapes associated latent vectors
Interpolating between file 1 and file 2

System arguments:
# Argument 1: Experiment name
# Argument 2: Either ssdf, gwn or udf
# Argument 3: Use either training data, test data or shape completion data
# Argument 4: Name of file 1 - only the filename and not the extension e.g. "name" and not "name.obj"
# Argument 5: Name of file 2 - only the filename and not the extension e.g. "name" and not "name.obj"
# Argument 6: Number of interpolations
"""

import os
import datetime
import shutil
import sys
from pygel3d import hmesh
#import igl
from dotenv import load_dotenv

# Deep Learning
#from __future__ import print_function, division
import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')

import numpy as np
array = np.array
from scipy.interpolate import interp1d

from utils.neural_network import data
from utils.neural_network import Net

from utils.general import parse_cfg_file

from utils.geometry import extract_mesh
from utils.geometry import gwn_grad_vector_ssdf
from utils.geometry import gwn_grad_vector_gwn
from utils.geometry import bisection_algorithm, project_algorithm
from utils.geometry import smooth_boundary_removal

def check_if_triangle_mesh(mesh):
    for f in mesh.faces():
        if (not len(mesh.circulate_face(f,mode='v')) == 3):
            return False
    return True

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
        "Plese indicate which type either ssdf, gwn, udf"
    
    assert len(sys.argv) > 3, \
        "Plese indicate either 'train' or 'test'"
    
    assert len(sys.argv) > 4, \
        "Plese indicate name of the first file without .obj extension"
    
    assert len(sys.argv) > 5, \
        "Plese indicate name of the second file without .obj extension"

    assert len(sys.argv) > 6, \
        "Plese provide the number of interpolations"
    
    if (len(os.sys.argv[4].split("."))==1):
        file1 = os.sys.argv[4] + ".obj"
    else:
        file1 = os.sys.argv[4]
    #else:
    #    file1 = os.sys.argv[4].split(".")[0]
    if (len(os.sys.argv[5].split("."))==1):
        file2 = os.sys.argv[5] + ".obj"
    else:
        #file2 = os.sys.argv[5].split(".")[0]
        file2 = os.sys.argv[5]

    # ---------------------------------------------------------------------------- #
    # Get environment file
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(".env"), \
        "Please create an .env file with appropriate directories"
    load_dotenv()

    # ---------------------------------------------------------------------------- #
    # mesh directories
    # ---------------------------------------------------------------------------- #
    mesh_train_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_train_dir'))
    mesh_test_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_test_dir'))
    mesh_shape_completion_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_shape_completion_dir'))

    # ---------------------------------------------------------------------------- #
    # model directories
    # ---------------------------------------------------------------------------- #
    model_train_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2] + '_model_train_dir'))
    model_test_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2]+'_model_test_dir'))
    model_shape_completion_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2]+'_model_shape_completion_dir'))

    # ---------------------------------------------------------------------------- #
    # interpolation directory
    # ---------------------------------------------------------------------------- #
    interpolation_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('interpolation_dir'))

    # ---------------------------------------------------------------------------- #
    # Load GWN threshold
    # ---------------------------------------------------------------------------- #
    if (sys.argv[2] != "udf"):
        path_gwn_threshold = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],"shape_reconstruction",os.sys.argv[2],"train","optimal_gwn_threshold.npy")
        assert (os.path.exists(path_gwn_threshold)), \
            "Optimal GWN threshold did not exist"
        gwn_threshold = float(np.load(path_gwn_threshold))

    # ---------------------------------------------------------------------------- #
    # configuration file
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
    # Obtain data
    # ---------------------------------------------------------------------------- #
    if (sys.argv[3] == "train"):
        mesh_data = data(mesh_train_dir, model_train_dir, cfg.model_train.name, device)
        save_path = os.path.join(os.getcwd(),interpolation_dir,"train",datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + os.sys.argv[2])
        checkpoint_latent_vector = torch.load(os.path.join(os.getcwd(),model_train_dir,cfg.model_train.name),map_location=device)
        model_nr_epoch_latent_vector = checkpoint_latent_vector['epoch']
        model_loss_latent_vector = checkpoint_latent_vector['loss']
    
    elif (sys.argv[3] == "test"):
        mesh_data = data(mesh_test_dir, model_test_dir, cfg.model_test.name, device)
        save_path = os.path.join(os.getcwd(),interpolation_dir,"test",datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + os.sys.argv[2])
        checkpoint_latent_vector = torch.load(os.path.join(os.getcwd(),model_test_dir,cfg.model_test.name),map_location=device)
        model_nr_epoch_latent_vector = checkpoint_latent_vector['epoch']
        model_loss_latent_vector = checkpoint_latent_vector['loss']
    
    else:
        assert False, \
            "Plese provide a name of a valid data set"
    
    # Create the directory
    os.makedirs(save_path)

    # ---------------------------------------------------------------------------- #
    # Neural network
    # ---------------------------------------------------------------------------- #
    if (sys.argv[2] == "ssdf"):
        network_output = cfg.network.multiple_output
    else:
        network_output = cfg.network.single_output

    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, network_output, device)

    checkpoint = torch.load(os.path.join(os.getcwd(),model_train_dir,cfg.model_train.name),map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    net = net.to(device)
    net.normalize_params()
    net.eval()

    model_nr_epoch = checkpoint['epoch']
    model_loss = checkpoint['loss']

    # ---------------------------------------------------------------------------- #
    # Status file
    # ---------------------------------------------------------------------------- #
    f = open(os.path.join(save_path,"Interpolations.txt"),"w")
    f.write("Interpolation file for shape interpolation\n")
    f.write("Parameters when doing shape interpolation\n")
    f.write("Using experiment: " + sys.argv[1] + "\n")
    f.write("Using method: " + sys.argv[2] + "\n")
    f.write("Using data set: " + sys.argv[3] + "\n")
    f.write("Interpolating between mesh " + file1 + " and mesh " + file2 + "\n")
    f.write("Using " + str(sys.argv[6]) + " number of interpolations\n")
    if (sys.argv[2] != "udf"):
        f.write("Using gwn_threshold: " + str(gwn_threshold) + " for gwn gradient computation\n")
    f.write("Number of smooth operations: " + str(cfg.surface_reconstruction.num_smooth) + "\n")
    f.write("Number of iterative projection operations: " + str(cfg.surface_reconstruction.num_project) + "\n")
    f.write("Number of bisection iterations: " + str(cfg.surface_reconstruction.num_bisection_iterations) + "\n")
    f.write("Increase the resolution with root3 subdivide by: " + str(cfg.surface_reconstruction.num_resolution_increase) + " number of times\n")
    f.write("Saving results in: " + save_path + "\n")
    f.write("The model has been trained for: " + str(model_nr_epoch) + " with loss: " + str(model_loss) + "\n")
    f.write("The model for latent vectors has been trained for: " + str(model_nr_epoch_latent_vector) + " with loss: " + str(model_loss_latent_vector) + "\n")
    f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    f.close()
    # Copy shape analysis script and configuration file
    shutil.copyfile(os.path.join(os.getcwd(),"interpolations.py"), os.path.join(save_path,"interpolations.py"))
    shutil.copyfile(cfg_file, os.path.join(save_path,"configuration_file.yaml"))

    # ---------------------------------------------------------------------------- #
    # Meshes to interolate between
    # ---------------------------------------------------------------------------- #
    f = open(os.path.join(save_path,"Interpolations.txt"),"a")
    f.write("Interpolating between mesh " + file1 + " and mesh " + file2 + "\n")
    f.close()

    mesh1, _, lv1 = mesh_data[file1]
    mesh2, _, lv2 = mesh_data[file2]
    nr_interpolations = int(sys.argv[6])

    # Latent vectors
    fst = lv1.detach().cpu().numpy()
    snd = lv2.detach().cpu().numpy()
    linfit = interp1d([1,nr_interpolations], np.vstack([fst, snd]), axis=0)

    mesh_resolution=cfg.surface_reconstruction.mesh_resolution
    dims = (mesh_resolution, mesh_resolution, mesh_resolution)
    bbox = np.array([[-1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0]])
    X = np.linspace(-1.0, 1.0, dims[0])
    Y = np.linspace(-1.0, 1.0, dims[1])
    Z = np.linspace(-1.0, 1.0, dims[2])
    P = np.array([array([X[idx[0]], Y[idx[1]], Z[idx[2]]]) for idx in np.ndindex(dims)])
    points3d = torch.from_numpy(P).float().to(device)

    for i in range(1,nr_interpolations+1):
        inter_lv = linfit(i)
        inter_lv = torch.from_numpy(inter_lv).float().to(device)
        net_input = torch.cat((inter_lv.repeat(len(P),1),points3d),1)
        
        with torch.no_grad():
            net_output = net.eval_forward(net_input)
        net_output_np = net_output.detach().cpu().numpy()

        if (os.sys.argv[2] == "ssdf"):
            net_SSDF = net_output_np[:,1].reshape(dims)

            # ---------------------------------------------------------------------------- #
            # Extract mesh
            # ---------------------------------------------------------------------------- #
            reconstructed_mesh = extract_mesh(device, model_train_dir, bbox, dims, net_SSDF, 
                                                    inter_lv, do_triangulation =False, 
                                                    num_smooth=cfg.surface_reconstruction.num_smooth, 
                                                    num_project=cfg.surface_reconstruction.num_project, 
                                                    iso_value=cfg.surface_reconstruction.iso_value)

            # ---------------------------------------------------------------------------- #
            # Stich and triangulate
            # ---------------------------------------------------------------------------- #
            hmesh.stitch(reconstructed_mesh)
            reconstructed_mesh.cleanup()
            hmesh.triangulate(reconstructed_mesh, clip_ear=True)
            if (not check_if_triangle_mesh(reconstructed_mesh)):
                hmesh.triangulate(reconstructed_mesh, clip_ear=False)

            # ---------------------------------------------------------------------------- #
            # Create a better triangulation
            # ---------------------------------------------------------------------------- #
            hmesh.minimize_dihedral_angle(reconstructed_mesh, max_iter=10000, anneal=True, alpha=False, gamma=4.0)
            hmesh.maximize_min_angle(reconstructed_mesh, dihedral_thresh=0.95, anneal=True)

            # ---------------------------------------------------------------------------- #
            # Increase resolution
            # ---------------------------------------------------------------------------- #
            for _ in range(cfg.surface_reconstruction.num_resolution_increase):
                hmesh.root3_subdivide(reconstructed_mesh)
                reconstructed_mesh = project_algorithm(device, model_train_dir, inter_lv, reconstructed_mesh, cfg.surface_reconstruction.num_project, cfg.surface_reconstruction.iso_value)

            # ---------------------------------------------------------------------------- #
            # Detect and remove holes
            # ---------------------------------------------------------------------------- #
            gwn_gradient = gwn_grad_vector_ssdf(device, model_train_dir, reconstructed_mesh, inter_lv)
            reconstructed_mesh = smooth_boundary_removal(reconstructed_mesh, gwn_gradient, gwn_threshold)

        
        elif (sys.argv[2] == "gwn"):
            net_GWN = net_output_np.reshape(dims)

            reconstructed_mesh = extract_mesh(device, model_train_dir, bbox, dims, net_GWN, 
                                                inter_lv, do_triangulation =False, 
                                                num_smooth=cfg.surface_reconstruction.num_smooth, 
                                                num_project=0, 
                                                iso_value=cfg.surface_reconstruction.iso_value, is_gwn=True)

            reconstructed_mesh = bisection_algorithm(device, model_train_dir, reconstructed_mesh, inter_lv, cfg.surface_reconstruction.num_bisection_iterations)

            # ---------------------------------------------------------------------------- #
            # Stich and triangulate
            # ---------------------------------------------------------------------------- #
            hmesh.stitch(reconstructed_mesh)
            reconstructed_mesh.cleanup()
            hmesh.triangulate(reconstructed_mesh, clip_ear=True)
            if (not check_if_triangle_mesh(reconstructed_mesh)):
                hmesh.triangulate(reconstructed_mesh, clip_ear=False)

            # ---------------------------------------------------------------------------- #
            # Create a better triangulation
            # ---------------------------------------------------------------------------- #
            hmesh.minimize_dihedral_angle(reconstructed_mesh, max_iter=10000, anneal=True, alpha=False, gamma=4.0)
            hmesh.maximize_min_angle(reconstructed_mesh, dihedral_thresh=0.95, anneal=True)

            # ---------------------------------------------------------------------------- #
            # Increase resolution
            # ---------------------------------------------------------------------------- #
            for _ in range(cfg.surface_reconstruction.num_resolution_increase):
                hmesh.root3_subdivide(reconstructed_mesh)
                reconstructed_mesh = bisection_algorithm(device, model_train_dir, reconstructed_mesh, inter_lv, cfg.surface_reconstruction.num_bisection_iterations)

            # ---------------------------------------------------------------------------- #
            # Detect and remove holes
            # ---------------------------------------------------------------------------- #
            gwn_gradient = gwn_grad_vector_gwn(device, model_train_dir, reconstructed_mesh, inter_lv)
            reconstructed_mesh = smooth_boundary_removal(reconstructed_mesh, gwn_gradient, gwn_threshold)

        elif (sys.argv[2] == "udf"):
            net_UDF = net_output_np.reshape(dims) 

            length = bbox[1][0] - bbox[0][0]
            width = bbox[1][1] - bbox[0][1]
            height = bbox[1][2] - bbox[0][2]

            L = max(length, max(width, height))

            net_UDF = np.multiply(net_UDF, 1.0/L) 

            bnfile = open(os.path.join(save_path, "interpolated_mesh_" + str(i)+  "_binary_unscaled.sdf"), "wb")
            bnfile.write(b'#sdf 1\n')
            bnfile.write(b'dim 150 150 150\n')
            bnfile.write(b'data\n')
            bnfile.write(bytearray(net_UDF))
            bnfile.close()

            f = open(os.path.join(save_path,"Interpolations.txt"),"a")
            f.write("Created binary file for " + "interpolated_mesh_" + str(i) + " using udf method\n")
            f.write("Bbox is: " + str(bbox) + "\n")
            f.write("length is: " + str(length) + "\n")
            f.write("width is: " + str(width) + "\n")
            f.write("height is: " + str(height) + "\n")
            f.write("L is: " + str(L) + "\n")
            f.write("1.0/L is: " + str(1.0/L) + "\n")
            f.write("With latent vector " + str(inter_lv) + "\n\n")
            f.close()

        if (sys.argv[2] != "udf"):
            hmesh.obj_save(os.path.join(save_path,"interpolated_mesh_"+str(i)+".obj"),reconstructed_mesh)
        f = open(os.path.join(save_path,"Interpolations.txt"),"a")
        f.write("Saved interpolated mesh: " + str(i) + "\n")
        f.close()
        
    f = open(os.path.join(save_path,"Interpolations.txt"),"a")
    f.write("Done with interpolation\n")
    f.close()
