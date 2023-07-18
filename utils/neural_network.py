"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Neural Network 

Purpose:
Functions used in the deep learning framework for the paper Neural Representation of Open Surfaces
"""

from pygel3d import hmesh
import igl

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as Func
import torch.nn.init as init
from torch.nn.parameter import Parameter

from bisect import bisect_left

import os
import numpy as np

# ------------------------------------------------------------------------------------------------------
# Latent Vector Class
# 
# mu:               float - The mean value of the normal distribution that the latent vectors are drawn from
# sigma:            float - The variance value of the normal distribution that the latent vectors are drawn from
# lv_size:          integer - the number of elements in the latent vector
# nr_meshes:        integer - the number of meshes
# device:           string - either 'cpu' or 'cuda'
#
# ------------------------------------------------------------------------------------------------------
class lv_class:
    def __init__(self, mu, sigma, lv_size, nr_meshes, device):
        self.mu = mu
        self.sigma = sigma
        self.latent_vector_size = lv_size
        self.latent_vectors = torch.normal(mu, sigma, size=(nr_meshes, self.latent_vector_size), requires_grad=True, device=device)     

    def __len__(self):
        return len(self.latent_vectors)    

# ------------------------------------------------------------------------------------------------------
# The neural network as a class
# 
# num_featuers:     integer - the input to the network
# num_hidden_units: numpy array - The number of hidden units for every layer
# num_output:       integer - the number of output nodes in the network
# prob_dropout:     numpy array - The dropout probability for every layer
#
# ------------------------------------------------------------------------------------------------------
class Net(nn.Module):

    def __init__(self, num_features, num_hidden_units, num_output, device):
        super().__init__()  
        
        # input layer
        self.W1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden_units[0], num_features))) # Kaiming initialization
        self.c1 = Parameter(torch.max(torch.sum(torch.abs(self.W1),axis=1)))
        self.b1 = Parameter(init.constant_(torch.Tensor(num_hidden_units[0]), 0))
        
        # 2nd layer
        self.W2 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden_units[1], num_hidden_units[0])))
        self.c2 = Parameter(torch.max(torch.sum(torch.abs(self.W2),axis=1)))
        self.b2 = Parameter(init.constant_(torch.Tensor(num_hidden_units[1]), 0))
        
        # 3rd layer
        self.W3 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden_units[2], num_hidden_units[1])))
        self.c3 = Parameter(torch.max(torch.sum(torch.abs(self.W3),axis=1)))
        self.b3 = Parameter(init.constant_(torch.Tensor(num_hidden_units[2]), 0))
        
        # 4th layer
        self.W4 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden_units[3]-num_features, num_hidden_units[2])))
        self.c4 = Parameter(torch.max(torch.sum(torch.abs(self.W4),axis=1)))
        self.b4 = Parameter(init.constant_(torch.Tensor(num_hidden_units[3]-num_features), 0))
        
        # 5th layer
        # Reintroduce the latent vector
        self.W5 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden_units[4], num_hidden_units[3])))
        self.c5 = Parameter(torch.max(torch.sum(torch.abs(self.W5),axis=1)))
        self.b5 = Parameter(init.constant_(torch.Tensor(num_hidden_units[4]), 0))
        
        # 6th layer
        self.W6 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden_units[5], num_hidden_units[4])))
        self.c6 = Parameter(torch.max(torch.sum(torch.abs(self.W6),axis=1)))
        self.b6 = Parameter(init.constant_(torch.Tensor(num_hidden_units[5]), 0))
        
        # 7th layer
        self.W7 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden_units[6], num_hidden_units[5])))
        self.c7 = Parameter(torch.max(torch.sum(torch.abs(self.W7),axis=1)))
        self.b7 = Parameter(init.constant_(torch.Tensor(num_hidden_units[6]), 0))

        # Final layer
        self.W8 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden_units[7])))
        self.c8 = Parameter(torch.max(torch.sum(torch.abs(self.W8),axis=1)))
        self.b8 = Parameter(init.constant_(torch.Tensor(num_output),0))
        
        # Activatiaon functions
        self.relu_activation = torch.nn.ReLU() # ReLU activation function
        self.hyp_tan_activation = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus()

        self.ones = torch.ones(1).to(device)
        
        
    def weight_normalization(self, W, softplus_c):
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(self.ones, softplus_c/absrowsum)
        W.data = W.data*scale[:,None]
        
    def train_forward(self, x_input):
        self.weight_normalization(self.W1, self.softplus(self.c1))
        x = Func.linear(x_input, self.W1, self.b1) # First layer
        x = self.relu_activation(x)        # The activitation function is set to the ReLU function.

        self.weight_normalization(self.W2, self.softplus(self.c2))
        x = Func.linear(x, self.W2, self.b2) # First layer
        x = self.relu_activation(x)              # Activation function

        self.weight_normalization(self.W3, self.softplus(self.c3))
        x = Func.linear(x, self.W3, self.b3) # Third layer
        x = self.relu_activation(x)              # Activation function

        self.weight_normalization(self.W4, self.softplus(self.c4))
        x = Func.linear(x, self.W4, self.b4) # Fourth layer
        x = self.relu_activation(x)              # Activation function

        self.weight_normalization(self.W5, self.softplus(self.c5))
        x = Func.linear(torch.cat((x, x_input), 1), self.W5, self.b5) # 5th layer
        x = self.relu_activation(x)              # Activation function

        self.weight_normalization(self.W6, self.softplus(self.c6))
        x = Func.linear(x, self.W6, self.b6) # 6th layer
        x = self.relu_activation(x)              # Activation function

        self.weight_normalization(self.W7, self.softplus(self.c7))
        x = Func.linear(x, self.W7, self.b7) # 7th layer
        x = self.relu_activation(x)              # Activation function

        self.weight_normalization(self.W8, self.softplus(self.c8))
        x = Func.linear(x, self.W8, self.b8) # Final layer
        x = self.hyp_tan_activation(x)
        return x
    
    def get_lipshitz_loss(self):
        loss_lip = self.ones
        loss_lip = torch.mul(loss_lip, self.softplus(self.c1))
        loss_lip = torch.mul(loss_lip, self.softplus(self.c2))
        loss_lip = torch.mul(loss_lip, self.softplus(self.c3))
        loss_lip = torch.mul(loss_lip, self.softplus(self.c4))
        loss_lip = torch.mul(loss_lip, self.softplus(self.c5))
        loss_lip = torch.mul(loss_lip, self.softplus(self.c6))
        loss_lip = torch.mul(loss_lip, self.softplus(self.c7))
        loss_lip = torch.mul(loss_lip, self.softplus(self.c8))
        return loss_lip
    
    def normalize_params(self):
        self.weight_normalization(self.W1, self.softplus(self.c1))
        self.weight_normalization(self.W2, self.softplus(self.c2))
        self.weight_normalization(self.W3, self.softplus(self.c3))
        self.weight_normalization(self.W4, self.softplus(self.c4))
        self.weight_normalization(self.W5, self.softplus(self.c5))
        self.weight_normalization(self.W6, self.softplus(self.c6))
        self.weight_normalization(self.W7, self.softplus(self.c7))
        self.weight_normalization(self.W8, self.softplus(self.c8))
        
    def eval_forward(self, x_input):
        # Dropout function and probability parameter p
        x = Func.linear(x_input, self.W1, self.b1) # First layer
        x = self.relu_activation(x)        # The activitation function is set to the ReLU function.

        x = Func.linear(x, self.W2, self.b2) # Second layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W3, self.b3) # Third layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W4, self.b4) # Fourth layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(torch.cat((x, x_input), 1), self.W5, self.b5) # 5th layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W6, self.b6) # 6th layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W7, self.b7) # 7th layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W8, self.b8) # Final layer
        x = self.hyp_tan_activation(x)
        return x

    def forward(self, x_input):
        # Dropout function and probability parameter p
        x = Func.linear(x_input, self.W1, self.b1) # First layer
        x = self.relu_activation(x)        # The activitation function is set to the ReLU function.

        x = Func.linear(x, self.W2, self.b2) # Second layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W3, self.b3) # Third layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W4, self.b4) # Fourth layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(torch.cat((x, x_input), 1), self.W5, self.b5) # 5th layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W6, self.b6) # 6th layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W7, self.b7) # 7th layer
        x = self.relu_activation(x)              # Activation function

        x = Func.linear(x, self.W8, self.b8) # Final layer
        x = self.hyp_tan_activation(x)
        return x
    

# ---------------------------------------------------------------------------- #
# Funciton for obtaining data
# ---------------------------------------------------------------------------- #
def obtain_data(filelist,mesh_dir):
    meshes = []
    triangle_cdf = []
    while True:
        next_line = filelist.readline().rstrip('\n')
        if next_line:
            obj_file = next_line + ".obj"
            triangle_areas = []
            meshes.append(obj_file)

            assert os.path.exists(os.path.join(os.getcwd(),mesh_dir, obj_file)), \
                "The obj_file " + obj_file + " did not exist"
            m = hmesh.obj_load(os.path.join(os.getcwd(),mesh_dir, obj_file))
            for f in m.faces():
                triangle_areas.append(m.area(f))

            # Using normal distribution
            triangle_areas = [tri_area / sum(triangle_areas) for tri_area in triangle_areas]
            triangle_areas = np.cumsum(triangle_areas)
            triangle_cdf.append(triangle_areas)

        if not next_line:
            break
    num_train_files = len(meshes)

    return meshes, num_train_files, triangle_cdf

# ---------------------------------------------------------------------------- #
# Class for handling sampling data for ssdf
# ---------------------------------------------------------------------------- #
class meshData_ssdf(Dataset):
    """Mesh dataset."""

    def __init__(self, mesh_list, root_dir, triangle_cdf,no_surface_points, no_box_points, sample_mean, sample_std1, sample_std2 ):
        """
        Args:
            mesh_list (string):     A .txt file providing all the names of the meshes
            root_dir (string):      Directory to the .txt providing all the names of the meshes
            triangle_cdf:           A list with length number of meshes. Every element is a list, that contains the accumulated sum of triangle
                                    areas divided by the total sum of triangle areas
            no_surface_points:      The number of point samples on the mesh
            no_box_points:          The number of points sampled in a bounding box of the mesh
            sample_mean:            The sample mean of the vectors sampled from a normal distribution
            sample_std1:            The standard variation of the uniformly generated 3x1 vector, which offsets the point sample from the surface
            sample_std2:            The standard variation of the other uniformly generated 3x1 vector, which offsets the point sample from the surface
        """
        self.mesh_list = mesh_list
        self.root_dir = root_dir
        mesh_distances = []
        meshes = []
        faces = []
        for i in range(len(mesh_list)):
            face_array = []
            m = hmesh.obj_load(os.path.join(self.root_dir, self.mesh_list[i]))
            mesh_distances.append(hmesh.MeshDistance(m))
            
            for f in m.faces():
                face_array.append(m.circulate_face(f,mode='v'))
            faces.append(face_array)
            meshes.append(m)
        
        self.meshes = meshes
        self.mesh_distances = mesh_distances
        self.faces = faces
        self.triangle_cdf = triangle_cdf
        
        self.num_surface_points = no_surface_points
        self.num_box_points = no_box_points
        self.mu_normal_vector = sample_mean
        self.sigma_normal_vector_1 = sample_std1
        self.sigma_normal_vector_2 = sample_std2
    

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        # So if you provide the index as a tensor, which you would do, when you sample,
        # then this is converted to a list.
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        # Not reading from disk
        mesh = self.meshes[idx]
        mesh_dist = self.mesh_distances[idx]
        
        face_array = self.faces[idx]
        
        #########################
        # points3d - 3D points
        #########################
        points3d = []
        
        # Comment to self: Maybe perform all computations in tensors, as this might be better performancewise
        # when using pytorch and dataloader. See this: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        # Bary centric coordinates
        [xi1,xi2] = torch.rand(2,self.num_surface_points)
        sqrt_xi1 = torch.sqrt(xi1)
        rn = [torch.sub(1.0,sqrt_xi1), torch.mul(torch.sub(1.0,xi2),sqrt_xi1), torch.mul(xi2,sqrt_xi1)]

        
        # For normal distribution
        tri_samples = torch.rand(self.num_surface_points).numpy()
        
        # Offset 3D point with normal vector
        normal_vector_1 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_1)).numpy()[0]
        normal_vector_2 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_2)).numpy()[0]

        for i in range(self.num_surface_points):
            
            # Get the triangle index corresponding to the random sample
            t_index = bisect_left(self.triangle_cdf[idx], tri_samples[i])
            
            # Sample bary-centric coordinates
            rn_select = np.array([rn[0][i],rn[1][i],rn[2][i]])
            [u,v,w] = rn_select[torch.randperm(3).numpy()]

            # Calculate position
            [v1,v2,v3] = np.array(mesh.circulate_face(t_index,mode='v'))
            point_p = u*mesh.positions()[v1] + v*mesh.positions()[v2] + w*mesh.positions()[v3]
            
            # Offset point
            points3d.append(point_p + normal_vector_1)
            points3d.append(point_p - normal_vector_1)
            points3d.append(point_p + normal_vector_2)
            points3d.append(point_p - normal_vector_2)

        # Sample box points
        x_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        y_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        z_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        box_points = torch.vstack((x_box,y_box,z_box)).t().numpy()

        points3d = np.array(points3d)
        points3d = np.vstack((points3d,box_points))

        ################################
        # udf - Unsigned Distance Field
        ################################
        udf = abs(mesh_dist.signed_distance(points3d)).reshape((len(points3d),1))
        udf = torch.from_numpy(udf).float()

        
        ##################################
        # gwn - generalized winding number
        ##################################
        gwn_shift = (2.0*(igl.fast_winding_number_for_meshes(np.array(mesh.positions()),np.array(face_array),
                                                            points3d)) - np.ones([1,len(points3d)])).T
        
        gwn_shift = torch.from_numpy(gwn_shift).float()

        ##################################
        # ssdf
        ##################################
        ssdf = torch.Tensor.mul(gwn_shift,udf)
  
        points3d = torch.from_numpy(points3d).float()
       
        index_vector = [idx]*(len(points3d))
        
        # Concatenate udf and gwn_udf
        target = torch.cat((torch.tanh(udf), torch.tanh(ssdf)), 1)
        
        sample = {'index_vector': index_vector, 'points3d': points3d, 'target': target}

        return sample

# ---------------------------------------------------------------------------- #
# Class for handling sampling data for ssdf shape completion
# ---------------------------------------------------------------------------- #
class meshData_ssdf_shape_completion(Dataset):
    """Mesh dataset."""

    def __init__(self, mesh_list, root_dir, triangle_cdf, no_surface_points, sample_mean, sample_std1, sample_std2):
        """
        Args:
            mesh_list (string):     A .txt file providing all the names of the meshes
            root_dir (string):      Directory to the .txt providing all the names of the meshes
            triangle_cdf:           A list with length number of meshes. Every element is a list, that contains the accumulated sum of triangle
                                    areas divided by the total sum of triangle areas
            no_surface_points:      The number of point samples on the mesh
            sample_mean:            The sample mean of the vectors sampled from a normal distribution
            sample_std1:            The standard variation of the uniformly generated 3x1 vector, which offsets the point sample from the surface
            sample_std2:            The standard variation of the other uniformly generated 3x1 vector, which offsets the point sample from the surface
        """
        self.mesh_list = mesh_list
        self.root_dir = root_dir
        mesh_distances = []
        meshes = []
        faces = []
        for i in range(len(mesh_list)):
            face_array = []
            m = hmesh.obj_load(os.path.join(self.root_dir, self.mesh_list[i]))
            mesh_distances.append(hmesh.MeshDistance(m))
            
            for f in m.faces():
                face_array.append(m.circulate_face(f,mode='v'))
            faces.append(face_array)
            meshes.append(m)
        
        self.meshes = meshes
        self.mesh_distances = mesh_distances
        self.faces = faces
        self.triangle_cdf = triangle_cdf
        
        self.num_surface_points = no_surface_points
        self.mu_normal_vector = sample_mean
        self.sigma_normal_vector_1 = sample_std1
        self.sigma_normal_vector_2 = sample_std2
    

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        # So if you provide the index as a tensor, which you would do, when you sample,
        # then this is converted to a list.
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        # Not reading from disk
        mesh = self.meshes[idx]
        mesh_dist = self.mesh_distances[idx]

        face_array = self.faces[idx]
        
        #########################
        # points3d - 3D points
        #########################
        points3d = []
        
        # Bary centric coordinates
        [xi1,xi2] = torch.rand(2,self.num_surface_points)
        sqrt_xi1 = torch.sqrt(xi1)
        rn = [torch.sub(1.0,sqrt_xi1), torch.mul(torch.sub(1.0,xi2),sqrt_xi1), torch.mul(xi2,sqrt_xi1)]
        
        # For normal distribution
        tri_samples = torch.rand(self.num_surface_points).numpy()
        
        # Offset 3D point with normal vector
        normal_vector_1 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_1)).numpy()[0]
        normal_vector_2 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_2)).numpy()[0]

        for i in range(self.num_surface_points):
            
            # Get the triangle index corresponding to the random sample
            t_index = bisect_left(self.triangle_cdf[idx], tri_samples[i])

            # Sample bary-centric coordinates
            rn_select = np.array([rn[0][i],rn[1][i],rn[2][i]])
            [u,v,w] = rn_select[torch.randperm(3).numpy()]

            # Calculate position
            [v1,v2,v3] = np.array(mesh.circulate_face(t_index,mode='v'))
            point_p = u*mesh.positions()[v1] + v*mesh.positions()[v2] + w*mesh.positions()[v3]
            
            # Offset point
            points3d.append(point_p + normal_vector_1)
            points3d.append(point_p - normal_vector_1)
            points3d.append(point_p + normal_vector_2)
            points3d.append(point_p - normal_vector_2)

        points3d = np.array(points3d)

        ################################
        # udf - Unsigned Distance Field
        ################################
        udf = abs(mesh_dist.signed_distance(points3d)).reshape((len(points3d),1))
        udf = torch.from_numpy(udf).float()

        
        ##################################
        # gwn - generalized winding number
        ##################################
        gwn_shift = (2.0*(igl.fast_winding_number_for_meshes(np.array(mesh.positions()),np.array(face_array),
                                                            points3d)) - np.ones([1,len(points3d)])).T
        
        gwn_shift = torch.from_numpy(gwn_shift).float()
        
        ##################################
        # gwn * udf
        ##################################
        ssdf = torch.Tensor.mul(gwn_shift,udf)
  
        points3d = torch.from_numpy(points3d).float()
       
        index_vector = [idx]*(len(points3d))
        
        # Concatenate udf and gwn_udf
        target = torch.cat((torch.tanh(udf), torch.tanh(ssdf)), 1)
        
        sample = {'index_vector': index_vector, 'points3d': points3d, 'target': target}

        return sample 

# ---------------------------------------------------------------------------- #
# Class for handling sampling data for gwn
# ---------------------------------------------------------------------------- #
class meshData_gwn(Dataset):
    """Mesh dataset."""

    def __init__(self, mesh_list, root_dir, triangle_cdf, no_surface_points, no_box_points, sample_mean, sample_std1, sample_std2):
        """
        Args:
            mesh_list (string):     A .txt file providing all the names of the meshes
            root_dir (string):      Directory to the .txt providing all the names of the meshes
            triangle_cdf:           A list with length number of meshes. Every element is a list, that contains the accumulated sum of triangle
                                    areas divided by the total sum of triangle areas
            no_surface_points:      The number of point samples on the mesh
            no_box_points:          The number of points sampled in a bounding box of the mesh
            sample_mean:            The sample mean of the vectors sampled from a normal distribution
            sample_std1:            The standard variation of the uniformly generated 3x1 vector, which offsets the point sample from the surface
            sample_std2:            The standard variation of the other uniformly generated 3x1 vector, which offsets the point sample from the surface
        """
        self.mesh_list = mesh_list
        self.root_dir = root_dir
        meshes = []
        faces = []
        for i in range(len(mesh_list)):
            face_array = []
            m = hmesh.obj_load(os.path.join(self.root_dir, self.mesh_list[i]))
            
            for f in m.faces():
                face_array.append(m.circulate_face(f,mode='v'))
            faces.append(face_array)
            meshes.append(m)
        
        self.meshes = meshes
        self.faces = faces
        self.triangle_cdf = triangle_cdf
        
        self.num_surface_points = no_surface_points
        self.num_box_points = no_box_points
        self.mu_normal_vector = sample_mean
        self.sigma_normal_vector_1 = sample_std1
        self.sigma_normal_vector_2 = sample_std2
    

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        # So if you provide the index as a tensor, which you would do, when you sample,
        # then this is converted to a list.
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        # Not reading from disk
        mesh = self.meshes[idx]
        
        face_array = self.faces[idx]
        
        #########################
        # points3d - 3D points
        #########################
        points3d = []
        
        # Comment to self: Maybe perform all computations in tensors, as this might be better performancewise
        # when using pytorch and dataloader. See this: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        # Bary centric coordinates
        [xi1,xi2] = torch.rand(2,self.num_surface_points)
        sqrt_xi1 = torch.sqrt(xi1)
        rn = [torch.sub(1.0,sqrt_xi1), torch.mul(torch.sub(1.0,xi2),sqrt_xi1), torch.mul(xi2,sqrt_xi1)]

        
        # For normal distribution
        tri_samples = torch.rand(self.num_surface_points).numpy()
        
        # Offset 3D point with normal vector
        normal_vector_1 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_1)).numpy()[0]
        normal_vector_2 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_2)).numpy()[0]

        for i in range(self.num_surface_points):
            
            # Get the triangle index corresponding to the random sample
            t_index = bisect_left(self.triangle_cdf[idx], tri_samples[i])
            
            # Sample bary-centric coordinates
            rn_select = np.array([rn[0][i],rn[1][i],rn[2][i]])
            [u,v,w] = rn_select[torch.randperm(3).numpy()]

            # Calculate position
            [v1,v2,v3] = np.array(mesh.circulate_face(t_index,mode='v'))
            point_p = u*mesh.positions()[v1] + v*mesh.positions()[v2] + w*mesh.positions()[v3]
            
            # Offset point
            points3d.append(point_p + normal_vector_1)
            points3d.append(point_p - normal_vector_1)
            points3d.append(point_p + normal_vector_2)
            points3d.append(point_p - normal_vector_2)

        # Sample box points
        x_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        y_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        z_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        box_points = torch.vstack((x_box,y_box,z_box)).t().numpy()

        points3d = np.array(points3d)
        points3d = np.vstack((points3d,box_points))
        
        ##################################
        # gwn - generalized winding number
        ##################################
        gwn_shift = (2.0*(igl.fast_winding_number_for_meshes(np.array(mesh.positions()),np.array(face_array),
                                                            points3d)) - np.ones([1,len(points3d)])).T
        
        target = torch.tanh(torch.from_numpy(gwn_shift).float())
        
        points3d = torch.from_numpy(points3d).float()
       
        index_vector = [idx]*(len(points3d))
        
        sample = {'index_vector': index_vector, 'points3d': points3d, 'target': target}

        return sample      

# ---------------------------------------------------------------------------- #
# Class for handling sampling data for gwn
# ---------------------------------------------------------------------------- #
class meshData_gwn_shape_completion(Dataset):
    """Mesh dataset."""

    def __init__(self, mesh_list, root_dir, triangle_cdf, no_surface_points, sample_mean, sample_std1, sample_std2):
        """
        Args:
            mesh_list (string):     A .txt file providing all the names of the meshes
            root_dir (string):      Directory to the .txt providing all the names of the meshes
            triangle_cdf:           A list with length number of meshes. Every element is a list, that contains the accumulated sum of triangle
                                    areas divided by the total sum of triangle areas
            no_surface_points:      The number of point samples on the mesh
            sample_mean:            The sample mean of the vectors sampled from a normal distribution
            sample_std1:            The standard variation of the uniformly generated 3x1 vector, which offsets the point sample from the surface
            sample_std2:            The standard variation of the other uniformly generated 3x1 vector, which offsets the point sample from the surface
        """
        self.mesh_list = mesh_list
        self.root_dir = root_dir
        meshes = []
        faces = []
        for i in range(len(mesh_list)):
            face_array = []
            m = hmesh.obj_load(os.path.join(self.root_dir, self.mesh_list[i]))
            hmesh.MeshDistance(m)
            
            for f in m.faces():
                face_array.append(m.circulate_face(f,mode='v'))
            faces.append(face_array)
            meshes.append(m)
        
        self.meshes = meshes
        self.faces = faces
        self.triangle_cdf = triangle_cdf
        
        self.num_surface_points = no_surface_points
        self.mu_normal_vector = sample_mean
        self.sigma_normal_vector_1 = sample_std1
        self.sigma_normal_vector_2 = sample_std2
    

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        # So if you provide the index as a tensor, which you would do, when you sample,
        # then this is converted to a list.
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        # Not reading from disk
        mesh = self.meshes[idx]
        
        face_array = self.faces[idx]
        
        #########################
        # points3d - 3D points
        #########################
        points3d = []
        
        # Bary centric coordinates
        [xi1,xi2] = torch.rand(2,self.num_surface_points)
        sqrt_xi1 = torch.sqrt(xi1)
        rn = [torch.sub(1.0,sqrt_xi1), torch.mul(torch.sub(1.0,xi2),sqrt_xi1), torch.mul(xi2,sqrt_xi1)]

        # For normal distribution
        tri_samples = torch.rand(self.num_surface_points).numpy()
        
        # Offset 3D point with normal vector
        normal_vector_1 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_1)).numpy()[0]
        normal_vector_2 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_2)).numpy()[0]

        for i in range(self.num_surface_points):
            
            # Get the triangle index corresponding to the random sample
            t_index = bisect_left(self.triangle_cdf[idx], tri_samples[i])
            
            # Sample bary-centric coordinates
            rn_select = np.array([rn[0][i],rn[1][i],rn[2][i]])
            [u,v,w] = rn_select[torch.randperm(3).numpy()]

            # Calculate position
            [v1,v2,v3] = np.array(mesh.circulate_face(t_index,mode='v'))
            point_p = u*mesh.positions()[v1] + v*mesh.positions()[v2] + w*mesh.positions()[v3]
            
            # Offset point
            points3d.append(point_p + normal_vector_1)
            points3d.append(point_p - normal_vector_1)
            points3d.append(point_p + normal_vector_2)
            points3d.append(point_p - normal_vector_2)

        points3d = np.array(points3d)

        ##################################
        # gwn - generalized winding number
        ##################################
        gwn_shift = (2.0*(igl.fast_winding_number_for_meshes(np.array(mesh.positions()),np.array(face_array),
                                                            points3d)) - np.ones([1,len(points3d)])).T
        
        target = torch.tanh(torch.from_numpy(gwn_shift).float())
        
  
        points3d = torch.from_numpy(points3d).float()
       
        index_vector = [idx]*(len(points3d))
        
        sample = {'index_vector': index_vector, 'points3d': points3d, 'target': target}

        return sample 

# ---------------------------------------------------------------------------- #
# Class for handling sampling data for udf
# ---------------------------------------------------------------------------- #
class meshData_udf(Dataset):
    """Mesh dataset."""

    def __init__(self, mesh_list, root_dir, triangle_cdf, no_surface_points, no_box_points, sample_mean, sample_std1, sample_std2):
        """
        Args:
            mesh_list (string):     A .txt file providing all the names of the meshes
            root_dir (string):      Directory to the .txt providing all the names of the meshes
            triangle_cdf:           A list with length number of meshes. Every element is a list, that contains the accumulated sum of triangle
                                    areas divided by the total sum of triangle areas
            no_surface_points:      The number of point samples on the mesh
            no_box_points:          The number of points sampled in a bounding box of the mesh
            sample_mean:            The sample mean of the vectors sampled from a normal distribution
            sample_std1:            The standard variation of the uniformly generated 3x1 vector, which offsets the point sample from the surface
            sample_std2:            The standard variation of the other uniformly generated 3x1 vector, which offsets the point sample from the surface
        """
        self.mesh_list = mesh_list
        self.root_dir = root_dir
        mesh_distances = []
        meshes = []
        for i in range(len(mesh_list)):
            m = hmesh.obj_load(os.path.join(self.root_dir, self.mesh_list[i]))
            mesh_distances.append(hmesh.MeshDistance(m))
            
            meshes.append(m)
        
        self.meshes = meshes
        self.mesh_distances = mesh_distances
        self.triangle_cdf = triangle_cdf
        
        self.num_surface_points = no_surface_points
        self.num_box_points = no_box_points
        self.mu_normal_vector = sample_mean
        self.sigma_normal_vector_1 = sample_std1
        self.sigma_normal_vector_2 = sample_std2
    
    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        # So if you provide the index as a tensor, which you would do, when you sample,
        # then this is converted to a list.
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        # Not reading from disk
        mesh = self.meshes[idx]
        mesh_dist = self.mesh_distances[idx]
        
        #########################
        # points3d - 3D points
        #########################
        points3d = []
        
        # Comment to self: Maybe perform all computations in tensors, as this might be better performancewise
        # when using pytorch and dataloader. See this: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        # Bary centric coordinates
        [xi1,xi2] = torch.rand(2,self.num_surface_points)
        sqrt_xi1 = torch.sqrt(xi1)
        rn = [torch.sub(1.0,sqrt_xi1), torch.mul(torch.sub(1.0,xi2),sqrt_xi1), torch.mul(xi2,sqrt_xi1)]

        # For normal distribution
        tri_samples = torch.rand(self.num_surface_points).numpy()
        
        # Offset 3D point with normal vector
        normal_vector_1 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_1)).numpy()[0]
        normal_vector_2 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_2)).numpy()[0]

        for i in range(self.num_surface_points):
            
            # Get the triangle index corresponding to the random sample
            t_index = bisect_left(self.triangle_cdf[idx], tri_samples[i])
            
            # Sample bary-centric coordinates
            rn_select = np.array([rn[0][i],rn[1][i],rn[2][i]])
            [u,v,w] = rn_select[torch.randperm(3).numpy()]

            # Calculate position
            [v1,v2,v3] = np.array(mesh.circulate_face(t_index,mode='v'))
            point_p = u*mesh.positions()[v1] + v*mesh.positions()[v2] + w*mesh.positions()[v3]
            
            # Offset point
            points3d.append(point_p + normal_vector_1)
            points3d.append(point_p - normal_vector_1)
            points3d.append(point_p + normal_vector_2)
            points3d.append(point_p - normal_vector_2)

        # Sample box points
        x_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        y_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        z_box = torch.sub(torch.mul(torch.rand(self.num_box_points),2.0),1.0)
        box_points = torch.vstack((x_box,y_box,z_box)).t().numpy()

        points3d = np.array(points3d)
        points3d = np.vstack((points3d,box_points))

        ################################
        # udf - Unsigned Distance Field
        ################################
        udf = abs(mesh_dist.signed_distance(points3d)).reshape((len(points3d),1))
        target = torch.tanh(torch.from_numpy(udf).float())

        points3d = torch.from_numpy(points3d).float()
       
        index_vector = [idx]*(len(points3d))
        
        sample = {'index_vector': index_vector, 'points3d': points3d, 'target': target}

        return sample    

# ---------------------------------------------------------------------------- #
# Class for handling sampling data for shape completion udf
# ---------------------------------------------------------------------------- #
class meshData_udf_shape_completion(Dataset):
    """Mesh dataset."""

    def __init__(self, mesh_list, root_dir, triangle_cdf, no_surface_points, sample_mean, sample_std1, sample_std2):
        """
        Args:
            mesh_list (string):     A .txt file providing all the names of the meshes
            root_dir (string):      Directory to the .txt providing all the names of the meshes
            triangle_cdf:           A list with length number of meshes. Every element is a list, that contains the accumulated sum of triangle
                                    areas divided by the total sum of triangle areas
            no_surface_points:      The number of point samples on the mesh
            sample_mean:            The sample mean of the vectors sampled from a normal distribution
            sample_std1:            The standard variation of the uniformly generated 3x1 vector, which offsets the point sample from the surface
            sample_std2:            The standard variation of the other uniformly generated 3x1 vector, which offsets the point sample from the surface
        """
        self.mesh_list = mesh_list
        self.root_dir = root_dir
        mesh_distances = []
        meshes = []
        for i in range(len(mesh_list)):
            m = hmesh.obj_load(os.path.join(self.root_dir, self.mesh_list[i]))
            mesh_distances.append(hmesh.MeshDistance(m))
            
            meshes.append(m)
        
        self.meshes = meshes
        self.mesh_distances = mesh_distances
        self.triangle_cdf = triangle_cdf
        
        self.num_surface_points = no_surface_points
        self.mu_normal_vector = sample_mean
        self.sigma_normal_vector_1 = sample_std1
        self.sigma_normal_vector_2 = sample_std2
    
    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        # So if you provide the index as a tensor, which you would do, when you sample,
        # then this is converted to a list.
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        
        # Not reading from disk
        mesh = self.meshes[idx]
        mesh_dist = self.mesh_distances[idx]
        
        #########################
        # points3d - 3D points
        #########################
        points3d = []
        
        # Comment to self: Maybe perform all computations in tensors, as this might be better performancewise
        # when using pytorch and dataloader. See this: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        # Bary centric coordinates
        [xi1,xi2] = torch.rand(2,self.num_surface_points)
        sqrt_xi1 = torch.sqrt(xi1)
        rn = [torch.sub(1.0,sqrt_xi1), torch.mul(torch.sub(1.0,xi2),sqrt_xi1), torch.mul(xi2,sqrt_xi1)]

        # For normal distribution
        tri_samples = torch.rand(self.num_surface_points).numpy()
        
        # Offset 3D point with normal vector
        normal_vector_1 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_1)).numpy()[0]
        normal_vector_2 = torch.normal(mean=self.mu_normal_vector, std=torch.empty(1,3).fill_(self.sigma_normal_vector_2)).numpy()[0]

        for i in range(self.num_surface_points):
            
            # Get the triangle index corresponding to the random sample
            t_index = bisect_left(self.triangle_cdf[idx], tri_samples[i])
            
            # Sample bary-centric coordinates
            rn_select = np.array([rn[0][i],rn[1][i],rn[2][i]])
            [u,v,w] = rn_select[torch.randperm(3).numpy()]

            # Calculate position
            [v1,v2,v3] = np.array(mesh.circulate_face(t_index,mode='v'))
            point_p = u*mesh.positions()[v1] + v*mesh.positions()[v2] + w*mesh.positions()[v3]
            
            # Offset point
            points3d.append(point_p + normal_vector_1)
            points3d.append(point_p - normal_vector_1)
            points3d.append(point_p + normal_vector_2)
            points3d.append(point_p - normal_vector_2)

        points3d = np.array(points3d)

        ################################
        # udf - Unsigned Distance Field
        ################################
        udf = abs(mesh_dist.signed_distance(points3d)).reshape((len(points3d),1))
        target = torch.tanh(torch.from_numpy(udf).float())

        points3d = torch.from_numpy(points3d).float()
       
        index_vector = [idx]*(len(points3d))
        
        sample = {'index_vector': index_vector, 'points3d': points3d, 'target': target}

        return sample      

#######################################################################################################
# Jupyter Notebook funcitons
#######################################################################################################

# ------------------------------------------------------------------------------------------------------
# Load the names of the meshes from the status file
# 
# data_path:    string - the directory to the data
#
# Returns filenames (python list) with the filenames
# ------------------------------------------------------------------------------------------------------
def load_data(data_path):
    filenames = []
    assert os.path.exists(os.path.join(data_path + "filelist.txt")), \
        "The filelist for " + str(data_path) + " does not exist"
    filelist = open(data_path + "filelist.txt", "r")
    while True:
        next_line = filelist.readline().rstrip('\n')
        if next_line:
            filenames.append(next_line + ".obj")
        if not next_line:
            break
    return filenames


# ------------------------------------------------------------------------------------------------------
# Load the mesh and the faces from the mesh
# 
# filename:     string - Filename of the mesh that should be loaded
# data_path:    string - the directory to the data
#
# Returns m (PyGEL3D mesh) and face_array (numpy array) with the vertices that make up every face
# ------------------------------------------------------------------------------------------------------
def load_mesh(filename,data_path):
    m = hmesh.obj_load(data_path + filename)
    face_array = []
    for f in m.faces():
        N = m.circulate_face(f,mode='v')
        face_array.append(N)
    return m, face_array


# ------------------------------------------------------------------------------------------------------
# Data class
# 
# data_dir:     string - Directory to the meshes (training meshes or test meshes)
# model_dir:    string - the directory to the model 
# model_name:   string - specify which model (best model or model at which epoch) to load
# device:       string - 'cpu' or 'cuda' where the model should be executed
#
# Can return the number of files
# Can return the mesh (PyGEL3D mesh), face_list (numpy array with vertices for every face), lv (tensor) 
# ------------------------------------------------------------------------------------------------------
class data:
    def __init__(self, data_dir, model_dir, model_name, device, test_data_dir=None):
        self.data_dir = data_dir
        self.filenames = load_data(data_dir)
        self.test_data_dir = test_data_dir
        if (not (test_data_dir is None)):
            self.test_filenames = load_data(test_data_dir)

        assert os.path.exists(os.path.join(os.getcwd(),model_dir,model_name)), \
            "The model " + str(os.path.join(os.getcwd(),model_dir,model_name)) + " does not exist. Please train the model or do inference"
        checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,model_name),map_location=device)
        self.latent_vectors = checkpoint['latent_vectors']
        self.latent_vectors.requires_grad = False

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # If index is an integer return the idx number mesh based on the filelist
        if (isinstance(idx, int)):
            assert idx < len(self.filenames), \
                "The index is greater that the number of files"
            mesh, face_list = load_mesh(self.filenames[idx],self.data_dir)
            lv = self.latent_vectors[idx]
            if (not (self.test_data_dir is None)):
                test_mesh, _ = load_mesh(self.test_filenames[idx],self.test_data_dir)

        # If index is a string return the mesh that has that specific filename
        elif (isinstance(idx, str)):
            assert idx in self.filenames, \
                "The requested file " + str(idx) + " is not a part of the data. Please check for spelling errors and that the fileformat is included"
            mesh, face_list = load_mesh(self.filenames[self.filenames.index(idx)],self.data_dir)
            lv = self.latent_vectors[self.filenames.index(idx)]
            if (not (self.test_data_dir is None)):
                test_mesh, _ = load_mesh(self.test_filenames[self.test_filenames.index(idx)],self.test_data_dir)
        hmesh.MeshDistance(mesh)

        if (self.test_data_dir is None):
            return mesh, face_list, lv
        else:
            return mesh, test_mesh, face_list, lv