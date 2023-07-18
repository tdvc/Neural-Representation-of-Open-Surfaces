# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Shape Reconstruction 

Purpose:
Reconstruct closed surfaces from the training, test or shape completion data set
"""

# Geometry libraries
from pygel3d import hmesh

# Deep Learning
#from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader

from utils.general import parse_cfg_file

from utils.neural_network import Net   

from utils.geometry import extract_mesh
from utils.geometry import bisection_algorithm, project_algorithm

# Math
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

# ---------------------------------------------------------------------------- #
# Class for returning input to the network
# ---------------------------------------------------------------------------- #
class meshData(Dataset):
    """Mesh dataset"""

    def __init__(self, mesh_list, root_dir, device, model_dir, model_name, mesh_resolution):
        
        self.root_dir = root_dir

        # Meshes
        meshes = []
        for i in range(len(mesh_list)):
            m = hmesh.obj_load(os.path.join(self.root_dir, mesh_list[i]))
            meshes.append(m)

        self.meshes = meshes
        self.mesh_list = mesh_list

        self.device = device

        self.mesh_resolution = mesh_resolution
        
        # Latent vectors
        checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,model_name),map_location=self.device)
        mesh_latent_vectors = checkpoint['latent_vectors']
        mesh_latent_vectors.requires_grad = False

        latent_vector_list = []
        for ii in range(len(mesh_list)):
            latent_vector_list.append(mesh_latent_vectors[ii])
        
        self.latent_vectors = latent_vector_list


    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        mesh_resolution=self.mesh_resolution
        dims = (mesh_resolution, mesh_resolution, mesh_resolution)
        bbox = np.array([[-1.0, -1.0, -1.0],
                        [1.0, 1.0, 1.0]])
        X = np.linspace(-1.0, 1.0, dims[0])
        Y = np.linspace(-1.0, 1.0, dims[1])
        Z = np.linspace(-1.0, 1.0, dims[2])
        
        P = np.array([array([X[index[0]], Y[index[1]], Z[index[2]]]) for index in np.ndindex(dims)])

        network_input = torch.cat((self.latent_vectors[idx].repeat(len(P),1).detach(),torch.from_numpy(P).float().to(self.device)),1)

        return network_input, bbox, dims, idx, self.mesh_list[idx], self.latent_vectors[idx]

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
        "Plese indicate which dataset 'train', 'test' or 'shape_completion' to use"

    assert (os.sys.argv[2] == "train" or os.sys.argv[2] == "test" or os.sys.argv[2] == "shape_completion"), \
        "Plese indicate which dataset 'train', 'test' or 'shape_completion' to use"

    assert len(os.sys.argv) > 3, \
        "Plese indicate which type either ssdf, gwn, udf"
    
    assert (os.sys.argv[3] == "ssdf" or os.sys.argv[3] == "gwn" or os.sys.argv[3] == "udf"), \
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
    if (not os.path.exists(shape_reconstruction_dir)):
        os.mkdir(shape_reconstruction_dir)

    if (os.sys.argv[2] == "train"):
        mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_train_dir'))
        model_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[3] + '_model_train_dir'))       
        model_name = cfg.model_train.name

    elif (os.sys.argv[2] == "test"):
        mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_test_dir'))
        model_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[3]+'_model_test_dir'))
        model_name = cfg.model_test.name

    else:
        mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_shape_completion_dir'))
        model_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[3]+'_model_shape_completion_dir'))
        model_name = cfg.model_shape_completion.name

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

    surface_mesh_resolution = cfg.surface_reconstruction.mesh_resolution

    meshDataset = meshData(mesh_files, mesh_dir, device, model_dir, model_name, 
                                    surface_mesh_resolution)

    # ---------------------------------------------------------------------------- #
    # Status file
    # ---------------------------------------------------------------------------- #
    status_file = os.path.join(shape_reconstruction_dir,cfg.status_file_shape_reconstruction)

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

    model_nr_epoch = checkpoint['epoch']
    model_loss = checkpoint['loss']
    
    # ---------------------------------------------------------------------------- #
    # Dataloaders
    # ---------------------------------------------------------------------------- #
    mesh_dataloader = DataLoader(meshDataset, batch_size=1, shuffle=False)
    mesh_data_iter = iter(mesh_dataloader)

    # ---------------------------------------------------------------------------- #
    # Write parameters to status file
    # ---------------------------------------------------------------------------- #
    if (os.path.exists(status_file)):
        f = open(status_file, "a")
    else:
        f = open(status_file, "w")
    f.write("Shape reconstruction for experiment " + os.sys.argv[1] + " using method " + os.sys.argv[3] + " for data set " + os.sys.argv[2] + "\n")
    f.write("Saving status file as " + status_file + "\n")
    f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n")
    f.write("Using parameters:\n")
    f.write("Number of smooth operations: " + str(cfg.surface_reconstruction.num_smooth) + "\n")
    f.write("Number of iterative projection operations: " + str(cfg.surface_reconstruction.num_project) + "\n")
    f.write("Number of bisection iterations: " + str(cfg.surface_reconstruction.num_bisection_iterations) + "\n")
    f.write("Increase the resolution with root3 subdivide by: " + str(cfg.surface_reconstruction.num_resolution_increase) + " number of times\n")
    f.write("Iso-contour value: " + str(cfg.surface_reconstruction.iso_value) + "\n")
    f.write("Mesh resolution: " + str(cfg.surface_reconstruction.mesh_resolution) + "\n") 
    f.write("The model has been trained for: " + str(model_nr_epoch) + " epochs with loss: " + str(model_loss) + "\n")
    f.write("The latent vectors have been trained for: " + str(model_nr_epoch_latent_vector) + " epochs with loss: " + str(model_loss_latent_vector) + "\n")
    f.close()
    # Copy shape analysis script and configuration file
    shutil.copyfile(os.path.join(os.getcwd(),"shape_reconstruction.py"), os.path.join(shape_reconstruction_dir,"shape_reconstruction.py"))
    shutil.copyfile(os.path.join(os.getcwd(),"utils","geometry.py"), os.path.join(shape_reconstruction_dir,"geometry.py"))
    shutil.copyfile(cfg_file, os.path.join(shape_reconstruction_dir,"configuration_file.yaml"))

    # ---------------------------------------------------------------------------- #
    # Loop over all meshes in the data set (either training or test determined by system arguments)
    # ---------------------------------------------------------------------------- #
    for i in range(len(mesh_dataloader)):
        net.eval()

        net_input, bbox, dims, idx, filename, latent_vector = mesh_data_iter.next()

        bbox = bbox.numpy()[0]

        # ---------------------------------------------------------------------------- #
        # Reconstruct ssdf mesh
        # ---------------------------------------------------------------------------- #
        if (os.sys.argv[3] == "ssdf"):

            if (not os.path.exists(os.path.join(shape_reconstruction_dir, filename[0].split(".")[0] + "_closed_reconstructed.obj"))):
                latent_vector = latent_vector[0]

                # reshape input
                net_input = net_input.view(-1,net_input.shape[2])

                with torch.no_grad():
                    net_output = net.eval_forward(net_input)
                net_output_np = net_output.detach().cpu().numpy()
                
                net_SSDF = net_output_np[:,1].reshape(dims)

                # ---------------------------------------------------------------------------- #
                # Extract mesh
                # ---------------------------------------------------------------------------- #
                reconstructed_mesh = extract_mesh(device, model_train_dir, bbox, dims, net_SSDF, 
                                                        latent_vector, do_triangulation =False, 
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
                    reconstructed_mesh = project_algorithm(device, model_train_dir, latent_vector, reconstructed_mesh, cfg.surface_reconstruction.num_project, cfg.surface_reconstruction.iso_value)

                hmesh.obj_save(os.path.join(shape_reconstruction_dir, filename[0].split(".")[0] + "_closed_reconstructed.obj"),reconstructed_mesh)

                f = open(status_file, "a")
                f.write("Reconstructed file " + filename[0].split(".")[0] + " using ssdf method\n")
                #f.write("With latent vector " + str(latent_vector) + "\n")
                f.close()   

                del net_input, bbox, dims, idx, filename, latent_vector, net_output, net_output_np, net_SSDF 

        # ---------------------------------------------------------------------------- #
        # Reconstruct gwn mesh
        # ---------------------------------------------------------------------------- #
        elif (os.sys.argv[3] == "gwn"):
            
            if (not os.path.exists(os.path.join(shape_reconstruction_dir, filename[0].split(".")[0] + "_closed_reconstructed.obj"))):
                latent_vector = latent_vector[0]

                # reshape input
                net_input = net_input.view(-1,net_input.shape[2])

                with torch.no_grad():
                    net_output = net.eval_forward(net_input)
                net_output_np = net_output.detach().cpu().numpy()
                
                
                net_GWN = net_output_np.reshape(dims)

                # ---------------------------------------------------------------------------- #
                # Extract mesh
                # ---------------------------------------------------------------------------- #
                gwn_mesh = extract_mesh(device, model_train_dir, bbox, dims, net_GWN, 
                                                        latent_vector, do_triangulation =False, 
                                                        num_smooth=cfg.surface_reconstruction.num_smooth, 
                                                        num_project=0, 
                                                        iso_value=cfg.surface_reconstruction.iso_value, is_gwn=True)

                reconstructed_mesh = bisection_algorithm(device, model_train_dir, gwn_mesh, latent_vector, cfg.surface_reconstruction.num_bisection_iterations)

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
                    reconstructed_mesh = bisection_algorithm(device, model_train_dir, reconstructed_mesh, latent_vector, cfg.surface_reconstruction.num_bisection_iterations)

                hmesh.obj_save(os.path.join(shape_reconstruction_dir, filename[0].split(".")[0] + "_closed_reconstructed.obj"),reconstructed_mesh)

                f = open(status_file, "a")
                f.write("Reconstructed file " + filename[0].split(".")[0] + " using gwn method\n")
                #f.write("With latent vector " + str(latent_vector) + "\n")
                f.close()

                del net_input, bbox, dims, idx, filename, latent_vector, net_output, net_output_np, net_GWN
        
        # ---------------------------------------------------------------------------- #
        # Reconstruct udf mesh using Neural Dual Contouring
        # ---------------------------------------------------------------------------- #
        elif (os.sys.argv[3] == "udf"):
            latent_vector = latent_vector[0]
        
            # reshape input
            net_input = net_input.view(-1,net_input.shape[2])

            with torch.no_grad():
                net_output = net.eval_forward(net_input)
            net_output_np = net_output.detach().cpu().numpy()
            
            net_UDF = net_output_np.reshape(dims) 

            length = bbox[1][0] - bbox[0][0]
            width = bbox[1][1] - bbox[0][1]
            height = bbox[1][2] - bbox[0][2]

            L = max(length, max(width, height))

            net_UDF = np.multiply(net_UDF, 1.0/L) 

            bnfile = open(os.path.join(shape_reconstruction_dir,filename[0].split(".")[0] + "_binary_unscaled.sdf"), "wb")
            bnfile.write(b'#sdf 1\n')
            bnfile.write(b'dim 150 150 150\n')
            bnfile.write(b'data\n')
            bnfile.write(bytearray(net_UDF))
            bnfile.close()

            f = open(status_file, "a")
            f.write("Created binary file " + filename[0].split(".")[0] + " using udf method\n")
            f.write("Bbox is: " + str(bbox) + "\n")
            f.write("length is: " + str(length) + "\n")
            f.write("width is: " + str(width) + "\n")
            f.write("height is: " + str(height) + "\n")
            f.write("L is: " + str(L) + "\n")
            f.write("1.0/L is: " + str(1.0/L) + "\n\n")
            #f.write("With latent vector " + str(latent_vector) + "\n")
            f.close()

            del net_input, bbox, dims, idx, filename, latent_vector, net_output, net_output_np

    f = open(status_file, "a")
    f.write("Endded reconstructing files\n")
    f.close()