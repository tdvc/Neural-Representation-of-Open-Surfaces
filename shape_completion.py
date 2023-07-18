# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Test 

Purpose:
Testing and optimizing latent vectors for test shapes
"""

# Deep Learning
#from __future__ import print_function, division
import torch
torch.multiprocessing.set_sharing_strategy('file_descriptor')
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

#from torch.distributions import Beta

# Math
import math
import numpy as np

# System libraries
import os
import datetime
from dotenv import load_dotenv
import shutil

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Utilities
from utils.general import parse_cfg_file
from utils.neural_network import Net
from utils.neural_network import lv_class
from utils.neural_network import obtain_data
from utils.neural_network import meshData_ssdf_shape_completion
from utils.neural_network import meshData_gwn_shape_completion
from utils.neural_network import meshData_udf_shape_completion

# ---------------------------------------------------------------------------- #
# Function for returning learning rate
# ---------------------------------------------------------------------------- #
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    print("Starting test for experiment " + os.sys.argv[1] +  " using ssdf")

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
        "Plese indicate either ssdf, gwn or udf"

    # ---------------------------------------------------------------------------- #
    # Get environment file
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(".env"), \
        "Please create an .env file with appropriate directories and specification of configuration file"
    load_dotenv()

    # ---------------------------------------------------------------------------- #
    # Set the directories for loading data, saving model and status file
    # ---------------------------------------------------------------------------- #
    mesh_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv('mesh_shape_completion_dir'))
    model_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2] + '_model_shape_completion_dir'))
    model_train_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2] + '_model_train_dir'))
    status_file_dir = os.path.join(os.getcwd(),"experiments",os.sys.argv[1],os.getenv(os.sys.argv[2] + '_shape_completion_status_file_dir'))

    # ---------------------------------------------------------------------------- #
    # Configuration file
    # ---------------------------------------------------------------------------- #
    cfg_file = os.path.join(os.getcwd(),"experiments", os.sys.argv[1],os.getenv('cfg_file'))
    assert os.path.exists(cfg_file), \
        "Make sure you have the config file (.yaml) in the cfgs folder and set correct path in .env file"
    cfg = parse_cfg_file(cfg_file)

    save_status_file = os.path.join(status_file_dir,cfg.status_file_test)
    save_model = os.path.join(model_dir,cfg.model_shape_completion.name)

    # ---------------------------------------------------------------------------- #
    # Load best result - lowest error
    # ---------------------------------------------------------------------------- #
    if (os.path.exists(os.path.join(model_dir,cfg.model_test.name))):
        checkpoint = torch.load(os.path.join(model_dir,cfg.model_test.name))
        lowest_error = checkpoint['loss']
    else:
        lowest_error = math.inf
    start_epoch = 0
        
    # ---------------------------------------------------------------------------- #
    # Set Status file
    # ---------------------------------------------------------------------------- #
    # Copy old status file
    if (cfg.model_test.load_best or cfg.model_test.load_from_epoch):
        if (os.path.exists(os.path.join(status_file_dir,cfg.status_file_test))):
            shutil.copyfile(os.path.join(status_file_dir,cfg.status_file_test), os.path.join(status_file_dir,cfg.status_file_test[:-4]+"_old.txt"))
    f = open(save_status_file, "w")
    f.write("Status file for shape completion for experiment " + os.sys.argv[1] + "\n")
    f.write("Saving status file in " + save_status_file + "\n")
    f.write("Saving trained models in " + save_model + "\n")
    f.write("Lowest error in test model is: " + str(lowest_error) + "\n")
    f.write("Running script at time: " + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "\n\n")
    f.close()
    # Copy shape completion script and configuration file
    shutil.copyfile(os.path.join(os.getcwd(),"shape_completion.py"), os.path.join(model_dir,"shape_completion.py"))
    shutil.copyfile(cfg_file, os.path.join(model_dir,"configuration_file.yaml"))

    # ---------------------------------------------------------------------------- #
    # Load data
    # ---------------------------------------------------------------------------- #
    assert os.path.exists(os.path.join(os.getcwd(),mesh_dir,"filelist.txt")), \
        "Please run preprocess_data.py first or create a filelist.txt with all the files"

    filelist = open(os.path.join(os.getcwd(),mesh_dir,"filelist.txt"), "r")

    meshes, num_test_files, triangle_cdf = obtain_data(filelist,mesh_dir)

    f = open(save_status_file,"a")
    f.write("Number of test meshes: " + str(num_test_files) + "\n")
    f.close()
    
    # ---------------------------------------------------------------------------- #
    # Write hyper parameters to status file
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    f.write("The hyper parameters are:" + "\n")

    f.write("Number_of_epochs: "+ str(cfg.testing.num_epochs) + "\n")
    f.write("Batch_size: "+ str(cfg.testing.batch_size) + "\n")
    f.write("Number of hidden units: "+ str(cfg.network.num_hidden_units) + "\n")
    f.write("Number_of_workers: "+ str(cfg.dataloader.num_workers_shape_completion) + "\n")

    f.write("Learning_rate_network: "+ str(cfg.network_optimizer.learning_rate) + "\n")

    f.write("Latent_vector_size: "+ str(cfg.latent_vector.size) + "\n")
    f.write("Latent_vector_mean: "+ str(cfg.latent_vector.mean) + "\n")
    f.write("Latent_vector_std: "+ str(cfg.latent_vector.std) + "\n")
    f.write("Learning_rate_latent_vectors: "+ str(cfg.lv_optimizer.learning_rate_train) + "\n")

    f.write("no_surface_points: "+ str(cfg.dataset.no_surface_points) + "\n")
    f.write("no_box_points: "+ str(0) + "\n")
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
        f.write("Using meshData_ssdf\n")
        f.close()
        meshDataset = meshData_ssdf_shape_completion(mesh_list = meshes, root_dir = mesh_dir, 
                                    triangle_cdf = triangle_cdf,
                                    no_surface_points=cfg.dataset.no_surface_points, 
                                    sample_mean=cfg.dataset.sample_mean, sample_std1=cfg.dataset.sample_std1,
                                    sample_std2=cfg.dataset.sample_std2 )

        network_output = cfg.network.multiple_output

    elif (os.sys.argv[2] == "gwn"):
        f = open(save_status_file,"a")
        f.write("Using meshData_gwn_shape_completion\n")
        f.close()
        meshDataset = meshData_gwn_shape_completion(mesh_list = meshes, root_dir = mesh_dir, triangle_cdf = triangle_cdf, no_surface_points=cfg.dataset.no_surface_points, sample_mean=cfg.dataset.sample_mean, sample_std1=cfg.dataset.sample_std1, sample_std2=cfg.dataset.sample_std2 )

        network_output = cfg.network.single_output

    elif (os.sys.argv[2] == "udf"):
        f = open(save_status_file,"a")
        f.write("Using meshData_udf\n")
        f.close()
        meshDataset = meshData_udf_shape_completion(mesh_list = meshes, root_dir = mesh_dir, 
                                    triangle_cdf = triangle_cdf,
                                    no_surface_points=cfg.dataset.no_surface_points,
                                    sample_mean=cfg.dataset.sample_mean, sample_std1=cfg.dataset.sample_std1,
                                    sample_std2=cfg.dataset.sample_std2 )

        network_output = cfg.network.single_output

    # ---------------------------------------------------------------------------- #
    # Data loader
    # ---------------------------------------------------------------------------- #
    set_num_workers = cfg.dataloader.num_workers_shape_completion
    if (not torch.cuda.is_available()):
        set_num_workers = 0
    dataloader = DataLoader(meshDataset, batch_size=cfg.testing.batch_size, shuffle=True, num_workers=set_num_workers)

    # ---------------------------------------------------------------------------- #
    # Set latent vectors
    # ---------------------------------------------------------------------------- #
    lv = lv_class(mu = cfg.latent_vector.mean, sigma=cfg.latent_vector.std, 
                        lv_size = cfg.latent_vector.size, nr_meshes= num_test_files, device=device)
    
    # ---------------------------------------------------------------------------- #
    # Load old checkpoint (Latent Vectors)
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    if (cfg.model_test.load_best):
        assert os.path.exists(os.path.join(model_dir,cfg.model_test.name)), \
            "No latent vectors have been optimized. Please test first"  
        checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_test.name),map_location=device)
        lv.latent_vectors = checkpoint['latent_vectors']
        f.write("Loaded best latent vectors")
    
    elif (cfg.model_test.load_from_epoch):
        assert os.path.exists(os.path.join(model_dir,cfg.model_test.name.split(".")[0] + "_epoch_" + cfg.model_test.load_from_epoch + ".pt")), \
            "No latent vectors have been optimized until epoch " + cfg.model_test.epoch_number 
        checkpoint = torch.load(os.path.join(model_dir,cfg.model_test.name.split(".")[0] + "_epoch_" + cfg.model_test.load_from_epoch + ".pt"),map_location=device)
        lv.latent_vectors = checkpoint['latent_vectors']
        f.write("Loaded latent vectors from epoch number " + cfg.model_test.epoch_number)
    f.close()

    # ---------------------------------------------------------------------------- #
    # Set network and load the best trained model
    # ---------------------------------------------------------------------------- #
    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, network_output, device)

    assert os.path.exists(os.path.join(model_train_dir,cfg.model_train.name)), \
            "No training model exists. Please train the model first before testing"
    checkpoint = torch.load(os.path.join(model_train_dir,cfg.model_train.name),map_location=device)

    train_model_nr_epoch = checkpoint['epoch']
    train_model_loss = checkpoint['loss']
    f = open(save_status_file,"a")
    f.write("The model has been trained for: " + str(train_model_nr_epoch) + " epochs with loss: " + str(train_model_loss) + "\n")
    f.close()
    net.load_state_dict(checkpoint['net_state_dict'])
    net = net.to(device)
    net.normalize_params()
    net.eval()

    # ---------------------------------------------------------------------------- #
    # Set optimizers and scheduler for reducing learning rate
    # ---------------------------------------------------------------------------- #
    latent_vector_optimizer = optim.Adam([lv.latent_vectors], lr=cfg.lv_optimizer.learning_rate_test) 
    L1_loss = nn.L1Loss(reduction='sum')
 
    # ---------------------------------------------------------------------------- #
    # Load old checkpoint (Latent vectors optimizer, Start epoch)
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    if (cfg.model_test.load_best):
        assert os.path.exists(os.path.join(os.getcwd(),model_dir,cfg.model_test.name)), \
            "No latent vectors have been optimized. Please test first"  
        checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_test.name),map_location='cpu')
        latent_vector_optimizer.load_state_dict(checkpoint['latent_vector_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        f.write("Loaded best latent vector optimizer")

    elif (cfg.model_test.load_from_epoch):
        assert os.path.exists(os.path.join(os.getcwd(),model_dir,cfg.model_test.name.split(".")[0] + "_epoch_" + cfg.model_test.epoch_number + ".pt")), \
            "No latent vectors have been optimized until epoch " + cfg.model_test.epoch_number 
        checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_test.name.split(".")[0] + "_epoch_" + cfg.model_test.epoch_number + ".pt"),map_location='cpu')
        latent_vector_optimizer.load_state_dict(checkpoint['latent_vector_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        f.write("Loaded latent vector optimizer from epoch number " + cfg.model_test.epoch_number)
    f.close()
    
    # ---------------------------------------------------------------------------- #
    # Test Loop
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    f.write("Starting from epoch: " + str(start_epoch+1) + "\n\n")
    f.close()

    losses = []
    
    for epoch in range((start_epoch+1),cfg.testing.num_epochs + (start_epoch+1)):
        cur_loss = 0.0
        
        net.eval()
        for i, batch in enumerate(dataloader):
            latent_vector_optimizer.zero_grad()
            
            # ---------------------------------------------------------------------------- #
            # Obtain network input from worker
            # ---------------------------------------------------------------------------- #
            batch_index_vector, batch_points3d, batch_target = batch['index_vector'], batch['points3d'], batch['target']
           
            batch_lv = lv.latent_vectors[np.reshape(np.vstack(batch_index_vector),(len(batch_index_vector)*len(batch_points3d)),order='F')]
            batch_points3d = batch_points3d.view(-1,3).to(device)
            batch_target = batch_target.view(-1,network_output).to(device)
            batch_input = torch.cat((batch_lv,batch_points3d),dim=1)
            
            # ---------------------------------------------------------------------------- #
            # Obtain network output
            # ---------------------------------------------------------------------------- #
            output = net.eval_forward(batch_input)
            output = output.to(device)
            
            # ---------------------------------------------------------------------------- #
            # Calulate cost
            # ---------------------------------------------------------------------------- #
            cost = L1_loss(output, batch_target)

            # ---------------------------------------------------------------------------- #
            # Compute Batch loss
            # ---------------------------------------------------------------------------- #
            #batch_loss = (cost/float(cfg.testing.batch_size))
            batch_loss = cost
            
            # ---------------------------------------------------------------------------- #
            # Compute gradients
            # ---------------------------------------------------------------------------- #
            batch_loss.backward()
            
            # ---------------------------------------------------------------------------- #
            # Optimize latent vector variables
            # ---------------------------------------------------------------------------- #
            latent_vector_optimizer.step()
            
            cur_loss += float(cost)

        losses.append(cur_loss)

        # ---------------------------------------------------------------------------- #
        # Frequency of printing progress
        # ---------------------------------------------------------------------------- #
        if (epoch % cfg.print_frequency == 0):
            f = open(save_status_file,"a")
            f.write("epoch: " + str(epoch) + " out of " + str(cfg.testing.num_epochs+start_epoch)  + "\n")
            f.write("Loss is: " + str(losses[-1]) + "\n\n")
            f.close()
        
        # ---------------------------------------------------------------------------- #
        # Frequency of saving model
        # ---------------------------------------------------------------------------- #
        if (epoch % cfg.save_model_frequency == 0):
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'latent_vectors': lv.latent_vectors,
                'latent_vector_optimizer_state_dict': latent_vector_optimizer.state_dict(),
                'loss': losses[-1],
            },os.path.join(os.getcwd(),model_dir,cfg.model_test.name.split(".")[0] + "_epoch_" + str(epoch) + ".pt"))
            f = open(save_status_file,"a")
            f.write("Saving model at epoch: " + str(epoch) + " out of " + str(cfg.testing.num_epochs+start_epoch)  + "\n")
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
                    'latent_vector_optimizer_state_dict': latent_vector_optimizer.state_dict(),
                    'loss': losses[-1],
                },save_model)
                f = open(save_status_file,"a")
                f.write("Saving best model at epoch: " + str(epoch) + " out of " + str(cfg.testing.num_epochs+start_epoch)  + "\n")
                f.write("Lowest error is: " + str(losses[-1]) + "\n\n")
                f.close()

    # ---------------------------------------------------------------------------- #
    # At the end of training loop save the last model
    # ---------------------------------------------------------------------------- #
    f = open(save_status_file,"a")
    if (losses[-1] < lowest_error):
        print("Final epoch: " + str(epoch) + " out of " + str(cfg.testing.num_epochs+start_epoch))
        f.write("Final epoch: " + str(epoch) + " out of " + str(cfg.testing.num_epochs+start_epoch)  + "\n")
        lowest_error = losses[-1]
        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'latent_vectors': lv.latent_vectors,
            'latent_vector_optimizer_state_dict': latent_vector_optimizer.state_dict(),
            'loss': losses[-1],
        },save_model)
    f.write("Finished Test\n")
    f.write("Lowest error is: " + str(lowest_error) + "\n")
    f.close() 
    
