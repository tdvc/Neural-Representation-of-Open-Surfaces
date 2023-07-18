# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Train 

Purpose:
Training network to predict GWN, UDF and SSDF
"""

# Geometry libraries
import igl
from pygel3d import hmesh

# Deep Learning
#from __future__ import print_function, division
import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Math
import math
import numpy as np

# System libraries
import os
from dotenv import load_dotenv
import shutil
import datetime 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Utilities
from utils.general import parse_cfg_file
from utils.neural_network import Net
from utils.neural_network import lv_class
from utils.neural_network import obtain_data
from utils.neural_network import meshData_ssdf
from utils.neural_network import meshData_gwn
from utils.neural_network import meshData_udf

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

    # ---------------------------------------------------------------------------- #
    # Get environment file
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(".env"), \
        "Please create an .env file with appropriate directories and specification of configuration file"
    load_dotenv()

    # ---------------------------------------------------------------------------- #
    # Set the directories for loading data, saving model and status file
    # ---------------------------------------------------------------------------- #
    mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_train_dir'))
    model_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2] + '_model_train_dir'))
    status_file_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('status_file_dir'),os.sys.argv[2])

    # ---------------------------------------------------------------------------- #
    # Configuration file
    # ---------------------------------------------------------------------------- #
    cfg_file = os.path.join(os.getcwd(),"experiments", os.sys.argv[1],os.getenv('cfg_file'))
    assert os.path.exists(cfg_file), \
        "Make sure you have the config file (.yaml) in the cfgs folder and set correct path in .env file"
    cfg = parse_cfg_file(cfg_file)

    save_status_file = os.path.join(status_file_dir,cfg.status_file_train)
    save_model = os.path.join(model_dir,cfg.model_train.name)

    # ---------------------------------------------------------------------------- #
    # Load best result - lowest error
    # ---------------------------------------------------------------------------- #
    if (os.path.exists(os.path.join(model_dir,cfg.model_train.name))):
        checkpoint = torch.load(os.path.join(model_dir,cfg.model_train.name),map_location='cpu')
        lowest_error = checkpoint['loss']
    else:
        lowest_error = math.inf
    start_epoch = 0

    # ---------------------------------------------------------------------------- #
    # Set Status file
    # ---------------------------------------------------------------------------- #
    # Copy old status file
    if (cfg.model_train.load_best or cfg.model_train.load_from_epoch):
        if (os.path.exists(os.path.join(os.getcwd(),status_file_dir,cfg.status_file_train))):
            shutil.copyfile(os.path.join(os.getcwd(),status_file_dir,cfg.status_file_train), os.path.join(status_file_dir,cfg.status_file_train[:-4]+"_old.txt"))
    f = open(save_status_file, "w")
    f.write("Status file for training for experiment "  + os.sys.argv[1] + " using " + os.sys.argv[2] +  "\n")
    f.write("Saving status file in " + save_status_file + "\n")
    f.write("Saving trained models in " + save_model + "\n")
    f.write("Lowest error in trained model is: " + str(lowest_error) + "\n")
    f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    f.close()
    # Copy training script and configuration file
    shutil.copyfile(os.path.join(os.getcwd(),"train.py"), os.path.join(model_dir,"train.py"))
    shutil.copyfile(os.path.join(os.getcwd(),"utils//neural_network.py"), os.path.join(model_dir,"neural_network.py"))
    shutil.copyfile(os.path.join(os.getcwd(),"utils//geometry.py"), os.path.join(model_dir,"geometry.py"))
    shutil.copyfile(cfg_file, os.path.join(model_dir,"configuration_file.yaml"))

    # ---------------------------------------------------------------------------- #
    # Load data
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(os.path.join(mesh_dir,"filelist.txt")), \
        "Please run preprocess_data.py first or create a filelist.txt with all the files"

    filelist = open(os.path.join(mesh_dir,"filelist.txt"), "r")

    meshes, num_train_files, triangle_cdf = obtain_data(filelist, mesh_dir)

    f = open(save_status_file,"a")
    f.write("Number of training meshes: " + str(num_train_files) + "\n\n")
    f.close()
    
    # ---------------------------------------------------------------------------- #
    # Write hyper parameters to status file
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    f.write("The hyper parameters are:" + "\n")
    f.write("Number_of_epochs: "+ str(cfg.training.num_epochs) + "\n")
    f.write("Batch_size: "+ str(cfg.training.batch_size) + "\n")
    f.write("Number of hidden units: "+ str(cfg.network.num_hidden_units) + "\n")
    f.write("Number_of_workers: "+ str(cfg.dataloader.num_workers) + "\n")

    f.write("Learning_rate_network: "+ str(cfg.network_optimizer.learning_rate) + "\n")
    
    f.write("Latent_vector_size: "+ str(cfg.latent_vector.size) + "\n")
    f.write("Latent_vector_mean: "+ str(cfg.latent_vector.mean) + "\n")
    f.write("Latent_vector_std: "+ str(cfg.latent_vector.std) + "\n")
    f.write("Learning_rate_latent_vectors: "+ str(cfg.lv_optimizer.learning_rate_train) + "\n")
    
    f.write("no_surface_points: "+ str(cfg.dataset.no_surface_points) + "\n")
    f.write("no_box_points: "+ str(cfg.dataset.no_box_points) + "\n")
    f.write("sample_mean: "+ str(cfg.dataset.sample_mean) + "\n")
    f.write("sample_std1: "+ str(cfg.dataset.sample_std1) + "\n")
    f.write("sample_std2: "+ str(cfg.dataset.sample_std2) + "\n\n")
    f.close()

    # ---------------------------------------------------------------------------- #
    # If GPU should be used check whether it is available
    # ---------------------------------------------------------------------------- #
    if (cfg.use_GPU):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    f = open(save_status_file,"a")
    if (not torch.cuda.is_available()):
        f.write("GPU is not available\n")
    else:
        f.write("GPU is available\n")
    f.write("Using device: " + str(device) + "\n")
    f.close()

    # ---------------------------------------------------------------------------- #
    # Data set
    # ---------------------------------------------------------------------------- #
    if (os.sys.argv[2] == "ssdf"):
        f = open(save_status_file,"a")
        f.write("Using meshData_ssdf \n")
        f.close()
        
        meshDataset = meshData_ssdf(mesh_list = meshes, root_dir = mesh_dir, 
                                    triangle_cdf = triangle_cdf,
                                    no_surface_points=cfg.dataset.no_surface_points, no_box_points=cfg.dataset.no_box_points,
                                    sample_mean=cfg.dataset.sample_mean, sample_std1=cfg.dataset.sample_std1,
                                    sample_std2=cfg.dataset.sample_std2 )

        network_output = cfg.network.multiple_output

    elif (os.sys.argv[2] == "gwn"):
        f = open(save_status_file,"a")
        f.write("Using meshData_gwn \n")
        f.close()

        meshDataset = meshData_gwn(mesh_list = meshes, root_dir = mesh_dir, 
                                    triangle_cdf = triangle_cdf,
                                    no_surface_points=cfg.dataset.no_surface_points, no_box_points=cfg.dataset.no_box_points,
                                    sample_mean=cfg.dataset.sample_mean, sample_std1=cfg.dataset.sample_std1,
                                    sample_std2=cfg.dataset.sample_std2 )

        network_output = cfg.network.single_output

    elif (os.sys.argv[2] == "udf"):
        f = open(save_status_file,"a")
        f.write("Using meshData_udf \n")
        f.close()

        meshDataset = meshData_udf(mesh_list = meshes, root_dir = mesh_dir, 
                                    triangle_cdf = triangle_cdf,
                                    no_surface_points=cfg.dataset.no_surface_points, no_box_points=cfg.dataset.no_box_points,
                                    sample_mean=cfg.dataset.sample_mean, sample_std1=cfg.dataset.sample_std1,
                                    sample_std2=cfg.dataset.sample_std2 )

        network_output = cfg.network.single_output
    
    # ---------------------------------------------------------------------------- #
    # Data loader
    # ---------------------------------------------------------------------------- #
    set_num_workers = cfg.dataloader.num_workers
    if (not torch.cuda.is_available()):
        set_num_workers = 0
    f = open(save_status_file,"a")
    f.write("num_workers is: " + str(set_num_workers) + "\n\n")
    f.close()
    dataloader = DataLoader(meshDataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=set_num_workers, pin_memory=True)


    # ---------------------------------------------------------------------------- #
    # Set network and latent vectors
    # ---------------------------------------------------------------------------- #
    # Latent vectors
    lv = lv_class(mu = cfg.latent_vector.mean, sigma=cfg.latent_vector.std, 
                        lv_size = cfg.latent_vector.size, nr_meshes= num_train_files, device=device)

    # Network
    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, network_output, device)

    # ---------------------------------------------------------------------------- #
    # Load old checkpoint (Network, Latent Vectors)
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    if (cfg.model_train.load_best):
        assert os.path.exists(os.path.join(model_dir,cfg.model_train.name)), \
            "No model has been trained. Please train first"  
        checkpoint = torch.load(os.path.join(model_dir,cfg.model_train.name),map_location=device)
        net.load_state_dict(checkpoint['net_state_dict'])
        lv.latent_vectors = checkpoint['latent_vectors']
        f.write("Loaded best network parameters and latent vectors\n")
    elif (cfg.model_train.load_from_epoch):
        assert os.path.exists(os.path.join(model_dir,cfg.model_train.name.split(".")[0] + "_epoch_" + cfg.model_train.epoch_number + ".pt")), \
            "No network has been trained until epoch " + cfg.model_train.epoch_number 

        checkpoint = torch.load(os.path.join(model_dir,cfg.model_train.name.split(".")[0] + "_epoch_" + cfg.model_train.epoch_number + ".pt"),map_location=device)
        net.load_state_dict(checkpoint['net_state_dict'])
        lv.latent_vectors = checkpoint['latent_vectors']
        f.write("Loaded network parameters, latent vectors and optimizer from epoch number " + cfg.model_train.epoch_number + "\n")
    f.close()
    net = net.to(device) 

    # ---------------------------------------------------------------------------- #
    # Set optimizers and scheduler for reducing learning rate
    # ---------------------------------------------------------------------------- #
    net_optimizer = optim.Adam(net.parameters(), lr=cfg.network_optimizer.learning_rate) # Without L2 regularization

    latent_vector_optimizer = optim.Adam([lv.latent_vectors], lr=cfg.lv_optimizer.learning_rate_train) # Without L2 regularization

    # ---------------------------------------------------------------------------- #
    # Load old checkpoint (Net optimizer, Latent vectors optimizer, Start epoch)
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    if (cfg.model_train.load_best):
        assert os.path.exists(os.path.join(model_dir,cfg.model_train.name)), \
            "No model has been trained. Please train first"  
        checkpoint = torch.load(os.path.join(model_dir,cfg.model_train.name),map_location='cpu')
        net_optimizer.load_state_dict(checkpoint['net_optimizer_state_dict'])
        latent_vector_optimizer.load_state_dict(checkpoint['latent_vector_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        f.write("Loaded best optimizers\n")

    elif (cfg.model_train.load_from_epoch):
        assert os.path.exists(os.path.join(model_dir,cfg.model_train.name.split(".")[0] + "_epoch_" + cfg.model_train.epoch_number + ".pt")), \
            "No network has been trained until epoch " + cfg.epoch_number 
        checkpoint = torch.load(os.path.join(model_dir,cfg.model_train.name.split(".")[0] + "_epoch_" + cfg.model_train.epoch_number + ".pt"),map_location='cpu')
        net_optimizer.load_state_dict(checkpoint['net_optimizer_state_dict'])
        latent_vector_optimizer.load_state_dict(checkpoint['latent_vector_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        f.write("Loaded optimizers from epoch number " + cfg.model_train.epoch_number + "\n")
    f.close()

    # ---------------------------------------------------------------------------- #
    # Loss function
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    f.write("Using ordinary loss\n\n")
    f.close()
    L1_loss = nn.L1Loss(reduction='sum')
        

    # ---------------------------------------------------------------------------- #
    # Training Loop
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    f.write("Starting from epoch: " + str(start_epoch+1) + "\n\n")
    f.close()
    
    losses = []

    for epoch in range((start_epoch+1), cfg.training.num_epochs + (start_epoch+1)):
        cur_loss = 0.0
        
        net.train()
        for i, batch in enumerate(dataloader):
            net_optimizer.zero_grad()
            latent_vector_optimizer.zero_grad()
            
            # ---------------------------------------------------------------------------- #
            # Obtain network input from worker
            # ---------------------------------------------------------------------------- #

            batch_index_vector, batch_points3d, batch_target = batch['index_vector'], batch['points3d'], batch['target']
            
            batch_lv = lv.latent_vectors[np.reshape(np.vstack(batch_index_vector),(len(batch_index_vector)*len(batch_points3d)),order='F')]

            batch_points3d = batch_points3d.view(-1,3).to(device)
            batch_target = batch_target.view(-1,network_output).to(device)
            batch_input = torch.cat((batch_lv,batch_points3d),dim=1).to(device)
            
            # ---------------------------------------------------------------------------- #
            # Obtain network output
            # ---------------------------------------------------------------------------- #
            output = net.train_forward(batch_input)
            output = output.to(device)

            # ---------------------------------------------------------------------------- #
            # Lipschitz loss
            # ---------------------------------------------------------------------------- #
            loss_lipschitz = net.get_lipshitz_loss()
            loss_lipschitz = loss_lipschitz.to(device)
                
            # ---------------------------------------------------------------------------- #
            # Calulate cost
            # ---------------------------------------------------------------------------- #
            batch_loss = L1_loss(output, batch_target) + cfg.network.alpha * loss_lipschitz
            
            # ---------------------------------------------------------------------------- #
            # Compute gradients
            # ---------------------------------------------------------------------------- #
            batch_loss.backward()
            
            # ---------------------------------------------------------------------------- #
            # Optimize network parameters and latent vector variables
            # ---------------------------------------------------------------------------- #
            net_optimizer.step()
            latent_vector_optimizer.step()

            # ---------------------------------------------------------------------------- #
            # Save loss
            # ---------------------------------------------------------------------------- #
            cur_loss += float(batch_loss)
            
        losses.append(cur_loss) # All batch losses

        # ---------------------------------------------------------------------------- #
        # Frequency of saving model
        # ---------------------------------------------------------------------------- #
        if (epoch % cfg.save_model_frequency == 0 or epoch == 1):
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'latent_vectors': lv.latent_vectors,
                'net_optimizer_state_dict': net_optimizer.state_dict(),
                'latent_vector_optimizer_state_dict': latent_vector_optimizer.state_dict(),
                'loss': losses[-1],
            },os.path.join(os.getcwd(),model_dir,cfg.model_train.name.split(".")[0] + "_epoch_" + str(epoch) + ".pt"))
            f = open(save_status_file,"a")
            f.write("Saving model at epoch: " + str(epoch) + " out of " + str(cfg.training.num_epochs+start_epoch)  + "\n")
            f.write("Loss is: " + str(losses[-1]) + "\n\n")
            f.close()
        
        # ---------------------------------------------------------------------------- #
        # Saving the best model
        # ---------------------------------------------------------------------------- #
        if (epoch % cfg.save_best_model_frequency == 0 or epoch == 1):
            if (losses[-1] < lowest_error):
                lowest_error = losses[-1]
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'latent_vectors': lv.latent_vectors,
                    'net_optimizer_state_dict': net_optimizer.state_dict(),
                    'latent_vector_optimizer_state_dict': latent_vector_optimizer.state_dict(),
                    'loss': losses[-1],
                },save_model)
                f = open(save_status_file,"a")
                f.write("Saving best model at epoch: " + str(epoch) + " out of " + str(cfg.training.num_epochs+start_epoch)  + "\n")
                f.write("Lowest error is: "  + str(losses[-1]) + "\n\n")
                f.close()
    
    # ---------------------------------------------------------------------------- #
    # At the end of training loop save the last model
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    if (losses[-1] < lowest_error):
        f.write("Final epoch: " + str(epoch) + " out of " + str(cfg.training.num_epochs+start_epoch)  + "\n")
        lowest_error = losses[-1]
        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'latent_vectors': lv.latent_vectors,
            'net_optimizer_state_dict': net_optimizer.state_dict(),
            'latent_vector_optimizer_state_dict': latent_vector_optimizer.state_dict(),
            'loss': losses[-1],
        },save_model)
    f.write("Finished Training\n")
    f.write("Lowest error is: " + str(lowest_error) + "\n\n")
    f.close() 
    
