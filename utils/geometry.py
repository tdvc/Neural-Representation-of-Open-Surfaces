# -*- coding: utf-8 -*-
"""
@author: Thor Vestergaard Christiansen (tdvc@dtu.dk)

Name: 
Geometry 

Purpose:
Geometry related functions used for the paper Neural Representation of Open Surfaces
"""

# ------------------------------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------------------------------

# Libraries for Geometry Processing
#import igl
from pygel3d import hmesh

# Math library
import numpy as np
import math
import random
array = np.array
import plotly.offline as py
import plotly.graph_objs as go

from bisect import bisect_left

# Deep learning library
import torch
from torch.autograd.functional import jacobian
if torch.cuda.is_available():
    print("cuda is avaiable")
from utils.neural_network import Net

from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator

# System library
import os
from dotenv import load_dotenv
from utils.general import parse_cfg_file

# ------------------------------------------------------------------------------------------------------
# Smooth a mesh by looping over each vertex in the mesh, compute the average of the neighbors positions 
# and use this as the vertex' position - essentially laplacian smoothing
# 
# Inputs:
# m:                PyGEL3D mesh 
# ------------------------------------------------------------------------------------------------------
def smooth(m):
    # """ Very simple mesh smoothing: computes new pos as average of neighbors"""
    pos = m.positions()
    new_pos = np.zeros(pos.shape)
    for v in m.vertices():
        for vn in m.circulate_vertex(v,mode='v'):
            new_pos[v] += pos[vn]
        new_pos[v] /= len(m.circulate_vertex(v,mode='v'))

    pos[:] = new_pos

# ------------------------------------------------------------------------------------------------------
# Checks whether a PyGEL3D mesh is a triangle mesh
# 
# Inputs:
# m:                PyGEL3D mesh 
# ------------------------------------------------------------------------------------------------------
def check_if_triangle_mesh(m):
    for f in m.faces():
        if (not len(m.circulate_face(f,mode='v')) == 3):
            return False
    return True

# ------------------------------------------------------------------------------------------------------
# Transform between a point grids' coordinate and the bounding box
#
# Inputs:
# Plow:                numpy array - The coordinate of the bounding box' lowest coordinate
# Phigh:               numpy array - The coordinate of the bounding box' highest coordinate
# dims:                tuple
# ------------------------------------------------------------------------------------------------------
class XForm:
    def __init__(self,Plow,Phigh,dims):

    # Initialize with the minimum coordinates corner Plow, maximum coordinates corner Phigh, and the voxel grid dimensions. """

        self.dims = array(dims)
        self.Plow = array(Plow)
        self.Phigh = array(Phigh)
        self.scaling = (self.Phigh - self.Plow) / self.dims
        self.inv_scaling = self.dims / (self.Phigh - self.Plow)

    def map(self,pi):
    #  map transforms a grid point to the point in space that corresponds to the grid point. """

        return self.scaling * array(pi) + self.Plow

    def inv_map(self,p):
        return array((p-self.Plow) * self.inv_scaling,dtype=int)

# ------------------------------------------------------------------------------------------------------
# Extract the surface from a volume grid with the learned SSDF predicted at every voxel grid point
#
# Inputs:
# device:           string - specify whether it is 'cpu' or 'cuda'
# model_dir:        string - directory to the stored network
# G:                numpy array - The point grid where each point is the network prediction of GWN*UDF
# xform:            oython class - transform between points
# latent_vector:    Pytorch tensor - The latent vector associated with the shape
# do_triangulation: boolean - specify whether the mesh should be triangulated or not
# num_smooth:       integer - specify how many times the mesh should be smoothed
# num_project:      integer - specify how many times the vertex should be projected on the iso-contour
# iso_value:        float - specify the iso-contour. Default is 0-level iso-contour
#
# Returns m (PyGEL3D mesh) 
# ------------------------------------------------------------------------------------------------------
def surface_ssdf(device, model_dir, G, xform, latent_vector, do_triangulation, num_smooth, num_project, iso=0.0):

    # Neural network
    load_dotenv()
    cfg_file = os.path.join(model_dir,"..","..","..",os.getenv('cfg_file'))
    cfg = parse_cfg_file(cfg_file)
    
    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, cfg.network.multiple_output, device)

    checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_train.name),map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.to(device)
    net.normalize_params()
    net.eval()

    # Create mesh and extract a rough surface
    m = hmesh.volumetric_isocontour(G, make_triangles=False, high_is_inside=True)

    # Stitch the vertices 
    hmesh.stitch(m)
    m.cleanup()
    pos = m.positions()

    # Transform the mesh to the scale of the point cloud
    for v in m.vertices():
        pos[v] = xform.map(pos[v])
    
    # Triangulate mesh
    if (do_triangulation):
        hmesh.triangulate(m)  

    # Project vertices onto 0 level isosurface
    for _ in range(num_smooth):
        smooth(m)

    # Create a new array with the positions
    new_pos = np.copy(pos)  

    for _ in range(num_project):
        net_input = torch.cat((latent_vector.repeat(len(pos),1),torch.from_numpy(np.array(pos)).to(device).float()),1)
        net_input.requires_grad_()
        net_input.to(device)
        
        net_output = net.eval_forward(net_input)
        net_output_np = net_output.cpu().detach().numpy()

        net_SSDF = net_output_np[:,1]

        for v in m.vertices():

            jacobi_matrix = jacobian(net, torch.reshape(net_input[v],(1,len(latent_vector)+3)))
            ssdf_grad = jacobi_matrix[0][1][0][-3:].cpu().detach().numpy()
            if (np.linalg.norm(ssdf_grad,2) > 1e-5):
                ssdf_grad_norm = ssdf_grad / np.linalg.norm(ssdf_grad,2)
                new_pos[v] = pos[v] - (net_SSDF[v] - iso) * ssdf_grad_norm / (np.linalg.norm(ssdf_grad_norm,2)**2.0)

        pos[:] = new_pos
        del net_SSDF, net_output, net_output_np, net_input
    return m

# ------------------------------------------------------------------------------------------------------
# Extract the surface from a volume grid with the learned GWN predicted at every voxel grid point
# 
# Inputs:
# G:                numpy array - The point grid where each point is the network prediction of GWN
# xform:            oython class - transform between points
# do_triangulation: boolean - specify whether the mesh should be triangulated or not
# num_smooth:       integer - specify how many times the mesh should be smoothed
#
# Returns m (PyGEL3D mesh) 
# ------------------------------------------------------------------------------------------------------
def surface_gwn(G, xform, do_triangulation, num_smooth):

    # Create mesh and extract a rough surface
    m = hmesh.volumetric_isocontour(G, make_triangles=False, high_is_inside=True)

    # Stitch the vertices 
    hmesh.stitch(m)
    m.cleanup()
    pos = m.positions()

    # Transform the mesh to the scale of the point cloud
    for v in m.vertices():
        pos[v] = xform.map(pos[v])
    
    # Triangulate mesh
    if (do_triangulation):
        hmesh.triangulate(m)  

    # Project vertices onto 0 level isosurface
    for _ in range(num_smooth):
        smooth(m)

    return m 

# ------------------------------------------------------------------------------------------------------
# Function that simply normalizes a vector
#
# v:                numpy array 
#
# Returns np array
# ------------------------------------------------------------------------------------------------------
def normalize_vector(v):
    return v / math.sqrt(np.dot(v,v))

# ------------------------------------------------------------------------------------------------------
# Function that simply computes the distance between two points
#
# v:                numpy array 
#
# Returns np array
# ------------------------------------------------------------------------------------------------------
def distance(p1,p2):
    return math.sqrt(np.dot(p1-p2,p1-p2))

# ------------------------------------------------------------------------------------------------------
# Function that projects points onto the iso-contour of the field.
#
# Inputs:
# device:           string - specify whether it is 'cpu' or 'cuda'
# model_dir:        string - directory to the stored network
# latent_vector:    Pytorch tensor - The latent vector associated with the shape
# num_project:      integer - specify how many times the vertex should be projected on the iso-contour
# iso_value:        float - specify the iso-contour. Default is 0-level iso-contour
#
# Returns np array
# ------------------------------------------------------------------------------------------------------
def project_algorithm(device, model_dir, latent_vector, mesh, num_project, iso=0.0):
    # Neural network
    load_dotenv()
    cfg_file = os.path.join(model_dir,"..","..","..",os.getenv('cfg_file'))
    cfg = parse_cfg_file(cfg_file)
    
    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, cfg.network.multiple_output, device)

    checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_train.name),map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.to(device)
    net.normalize_params()
    net.eval()

    m = hmesh.Manifold(mesh)
    pos = m.positions()

    # Create a new array with the positions
    new_pos = np.copy(pos)  

    for _ in range(num_project):
        net_input = torch.cat((latent_vector.repeat(len(pos),1),torch.from_numpy(np.array(pos)).to(device).float()),1)
        net_input.requires_grad_()
        net_input.to(device)
        
        net_output = net.eval_forward(net_input)
        net_output_np = net_output.cpu().detach().numpy()

        net_SSDF = net_output_np[:,1]

        for v in m.vertices():

            jacobi_matrix = jacobian(net, torch.reshape(net_input[v],(1,len(latent_vector)+3)))
            ssdf_grad = jacobi_matrix[0][1][0][-3:].cpu().detach().numpy()
            if (np.linalg.norm(ssdf_grad,2) > 1e-5):
                ssdf_grad_norm = ssdf_grad / np.linalg.norm(ssdf_grad,2)
                new_pos[v] = pos[v] - (net_SSDF[v] - iso) * ssdf_grad_norm / (np.linalg.norm(ssdf_grad_norm,2)**2.0)

        pos[:] = new_pos
        del net_SSDF, net_output, net_output_np, net_input
    return m

# ------------------------------------------------------------------------------------------------------
# Function that uses bisection to place points on to the zero-level iso-contour
#
# Inputs:
# device:           string - specify whether it is 'cpu' or 'cuda'
# model_dir:        string - directory to the stored network
# mesh:             PYGEL3D mesh - the mesh entity
# latent_vector:    Pytorch tensor - The latent vector associated with the shape
# iteration:        integer - specify the numbrer of iterations of the bisection algorithm
#
# Returns rm (PyGEL3D mesh) 
# ------------------------------------------------------------------------------------------------------
def bisection_algorithm(device, model_dir, mesh, latent_vector, iteration):

    # Neural network
    load_dotenv()
    cfg_file = os.path.join(model_dir,"..","..","..",os.getenv('cfg_file'))
    cfg = parse_cfg_file(cfg_file)
        
    net = Net(cfg.latent_vector.size + cfg.network.size_point, cfg.network.num_hidden_units, 1, device)

    checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_train.name),map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.to(device)
    net.normalize_params()
    net.eval()

    # Mesh 
    rm = hmesh.Manifold(mesh)
    
    pos = rm.positions()
    k = hmesh.average_edge_length(rm)

    vns = np.zeros((rm.no_allocated_vertices(),3))
    for v in rm.vertices():
        vns[v] = rm.vertex_normal(v)
        
    scale = 0.01
    pas = pos + k * scale * vns
    pbs = pos - k * scale * vns
    
    while True:
        net_input = torch.cat((latent_vector.repeat(len(pas),1).detach(),torch.from_numpy(pas).float().to(device)),1)
        with torch.no_grad():
            net_output = net.eval_forward(net_input)
            pas_values = net_output.detach().cpu().numpy().reshape(len(pas))
        if (sum(pas_values >= 0.0) == 0 or 3.0 - scale < 1e-3):
            break
        scale += 0.01
        pas[pas_values >= 0.0] = pos[pas_values >= 0.0] + k * scale * vns[pas_values >= 0.0]
    
    scale = 0.01
    while True:
        net_input = torch.cat((latent_vector.repeat(len(pbs),1).detach(),torch.from_numpy(pbs).float().to(device)),1)
        with torch.no_grad():
            net_output = net.eval_forward(net_input)
            pbs_values = net_output.detach().cpu().numpy().reshape(len(pbs))
        if (sum(pbs_values < 0.0) == 0 or 3.0 - scale < 1e-3):
            break
        scale += 0.01

        pbs[pbs_values < 0.0] = pos[pbs_values < 0.0] - k * scale * vns[pbs_values < 0.0]
  
    changed_coordinates = np.multiply(pas_values <= 0.0, pbs_values > 0.0 )
 
    net_input =  torch.cat((latent_vector.repeat(len(pas),1).detach(),torch.from_numpy(pas).float().to(device)),1)
    with torch.no_grad():
        net_output = net.eval_forward(net_input)
        pas_values = net_output.detach().cpu().numpy().reshape(len(pas))

    net_input = torch.cat((latent_vector.repeat(len(pbs),1).detach(),torch.from_numpy(pbs).float().to(device)),1)
    with torch.no_grad():
        net_output = net.eval_forward(net_input)
        pbs_values = net_output.detach().cpu().numpy().reshape(len(pbs))

    for _ in range(iteration):
        d = np.linalg.norm(pas-pbs,2,axis=1)/2.0
        pcs = pbs + np.multiply(np.tile(d,(3,1)).T,vns)

        net_input = torch.cat((latent_vector.repeat(len(pcs),1).detach(),torch.from_numpy(pcs).float().to(device)),1)
        with torch.no_grad():
            net_output = net.eval_forward(net_input)
            pcs_values = net_output.detach().cpu().numpy().reshape(len(pcs))

        pbs[pcs_values > 0.0] = pcs[pcs_values > 0.0]
        pas[pcs_values <= 0.0] = pcs[pcs_values <= 0.0]
    
    pos[changed_coordinates] = pbs[changed_coordinates]
    
    return rm

# ------------------------------------------------------------------------------------------------------
# Compute the gradient of the Generalized Winding Number at every vertex on the mesh
# OBS: Only for the network that outputs both UDF and GWN*UDF
#
# Inputs: 
# device:           string - specify whether it is 'cpu' or 'cuda'
# model_dir:        string - directory to the stored network
# m:                PyGEL3D mesh - the extracted mesh
# latent_vector:    Pytorch tensor - The latent vector associated with the shape
#
# Returns gwn_grad_vector (numpy array)
# ------------------------------------------------------------------------------------------------------
def gwn_grad_vector_ssdf(device, model_dir, m, latent_vector):

    # Neural network
    load_dotenv()
    cfg_file = os.path.join(model_dir,"..","..","..",os.getenv('cfg_file'))
    cfg = parse_cfg_file(cfg_file)

    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, cfg.network.multiple_output, device)
    
    checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_train.name),map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.to(device)
    net.normalize_params()
    net.eval()

    if (isinstance(m, (list, tuple, np.ndarray))):
        pos = m
    else:
        pos = m.positions()
    net_input = torch.cat((latent_vector.repeat(len(pos),1).detach(),torch.from_numpy(np.array(pos)).float().to(device)),1)

    net_input.requires_grad_()
    net_input.to(device)
    net_output = net.eval_forward(net_input)
    output_gwn = torch.div(torch.add(torch.div(net_output[:,1],abs(net_output[:,0])+1e-5),1.0),2.0)
    output_gwn.retain_grad()
    external_grad = torch.tensor(len(net_input)*[1.]).to(device)
    
    output_gwn.backward(gradient=external_grad)

    gwn_grad_vector = np.zeros(len(net_input.grad[:,-3:]))
    for i in range(len(net_input.grad[:,-3:])):
        gwn_grad_vector[i] = np.linalg.norm(net_input.grad[i,-3:].cpu().numpy(),2)
    return gwn_grad_vector

# ------------------------------------------------------------------------------------------------------
# Compute the gradient of the Generalized Winding Number at every vertex on the mesh
# OBS: Only for the network that only has one output
#
# device:           string - specify whether it is 'cpu' or 'cuda'
# model_dir:        string - directory to the stored network
# m:                PyGEL3D mesh - the extracted mesh
# latent_vector:    Pytorch tensor - The latent vector associated with the shape
#
# Returns gwn_grad_vector (numpy array)
# ------------------------------------------------------------------------------------------------------
def gwn_grad_vector_gwn(device, model_dir, m, latent_vector):

    # Neural network
    load_dotenv()
    cfg_file = os.path.join(model_dir,"..","..","..",os.getenv('cfg_file'))
    cfg = parse_cfg_file(cfg_file)
    
    net = Net(cfg.latent_vector.size + cfg.network.size_point, 
                cfg.network.num_hidden_units, cfg.network.single_output, device)
    
    checkpoint = torch.load(os.path.join(os.getcwd(),model_dir,cfg.model_train.name),map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.to(device)
    net.normalize_params()
    net.eval()

    if (isinstance(m, (list, tuple, np.ndarray))):
        pos = m
    else:
        pos = m.positions()
    net_input = torch.cat((latent_vector.repeat(len(pos),1).detach(),torch.from_numpy(np.array(pos)).float().to(device)),1)

    net_input.requires_grad_()
    net_input.to(device)
    output_gwn = net.eval_forward(net_input)
    output_gwn = torch.div(torch.add(output_gwn,1.0),2.0)
    output_gwn.retain_grad()
    external_grad = torch.tensor(len(net_input)*[[1.]]).to(device)
    
    output_gwn.backward(gradient=external_grad)

    gwn_grad_vec = np.zeros(len(net_input.grad[:,-3:]))
    for i in range(len(net_input.grad[:,-3:])):
        gwn_grad_vec[i] = np.linalg.norm(net_input.grad[i,-3:].cpu().numpy(),2)
    return gwn_grad_vec

# ------------------------------------------------------------------------------------------------------
# Extract mesh using GWN*UDF
#
# device:           string - specify whether it is 'cpu' or 'cuda'
# model_dir:        string - directory to the stored network
# bbox:             numpy array - approximate bounding box for the mesh - Found using bbox_dims function
# dims:             tuple - The bounding box of the mesh divided into equally sized cubes
# net_GWN_UDF:      numpy array - The point grid where each point is the network prediction of GWN*UDF
# latent_vector:    Pytorch tensor - The latent vector associated with the shape
# do_triangulation: boolean - specify whether the mesh should be triangulated or not
# num_smooth:       integer - specify how many times the mesh should be smoothed
# num_project:      integer - specify how many times the vertex should be projected on the iso-contour
# iso_value:        float - specify the iso-contour. Default is 0-level iso-contour
# is_gwn:           boolean - to specify which surface extraction function should be used
#
# Returns mesh (PyGEL3D mesh)
# ------------------------------------------------------------------------------------------------------
def extract_mesh(device, model_dir, bbox, dims, vol_grid, latent_vector, do_triangulation, num_smooth, num_project, iso_value, is_gwn=None):
    print(bbox)
    xform = XForm(bbox[0], bbox[1], dims)

    if (is_gwn):
        mesh = surface_gwn(vol_grid, xform, do_triangulation, num_smooth)
    else:
        mesh = surface_ssdf(device, model_dir, vol_grid, xform, latent_vector, do_triangulation, num_smooth, num_project, iso=iso_value)

    return mesh

# ------------------------------------------------------------------------------------------------------
# Compute the chamfer distance and the mesh accuracy between two meshes using meshwise approach
#
# gtm:                              PyGEL3D mesh - ground truth mesh
# rm:                               PyGEL3D mesh - reconstructed mesh
# number_of_points_chamfer:         int - number of points used to compute chamfer distance
# number_of_points_mesh_accuracy:   int - number of points used to compute mesh accuracy
# filepath:                         string - A path to the directory where the points sampled on the ground truth mesh is s tored
# filename:                         string - Name of the file for which we sample points
#
# Returns float
# ------------------------------------------------------------------------------------------------------
def calculate_chamfer_distance_and_mesh_accuracy(gtm, rm, number_of_points_chamfer, number_of_points_mesh_accuracy, filepath, filename):
    # ---------------------------------------------------------------------------- #
    # Sample points on ground truth mesh
    # ---------------------------------------------------------------------------- #
    gtm_sampled_points = os.path.join(filepath,filename + "_gtm_sampled_points_chamfer_distance.npy")

    if (not os.path.exists(gtm_sampled_points)):
        # gtm arrays 
        gtm_triangle_areas = []
        for f in gtm.faces():
            gtm_triangle_areas.append(gtm.area(f))

        gtm_triangle_areas = [tri_area / sum(gtm_triangle_areas) for tri_area in gtm_triangle_areas]
        gtm_triangle_cdf = np.cumsum(gtm_triangle_areas)
        
        # ---------------------------------------------------------------------------- #
        # Sample points on gtm
        # ---------------------------------------------------------------------------- #
        gtm_tri_samples = torch.rand(number_of_points_chamfer).numpy()
        
        gtm_ran_var = torch.rand(2,number_of_points_chamfer).numpy()

        points_gtm = []
        for i in range(number_of_points_chamfer):
            [xi1,xi2] = gtm_ran_var[:,i]
            sqrt_xi1 = math.sqrt(xi1)
            rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
            rn_select = torch.randperm(3).numpy()
            u = rn[rn_select[0]]
            v = rn[rn_select[1]]
            w = rn[rn_select[2]]

            t_index = bisect_left(gtm_triangle_cdf, gtm_tri_samples[i])
       
            [v1,v2,v3] = np.array(gtm.circulate_face(t_index,mode='v'))

            points_gtm.append(u*gtm.positions()[v1] + v*gtm.positions()[v2] + w*gtm.positions()[v3])
        points_gtm = np.array(points_gtm)

        np.save(gtm_sampled_points,points_gtm)

    points_gtm = np.load(gtm_sampled_points)

    # ---------------------------------------------------------------------------- #
    # Sample points on rm
    # ---------------------------------------------------------------------------- #
    # rm arrays
    rm_triangle_areas = []
    for f in rm.faces():
        rm_triangle_areas.append(rm.area(f))

    rm_triangle_areas = [rm_tri_area / sum(rm_triangle_areas) for rm_tri_area in rm_triangle_areas]
    rm_triangle_cdf = np.cumsum(rm_triangle_areas)

    rm_tri_samples = torch.rand(number_of_points_chamfer).numpy()
    
    rm_ran_var = torch.rand(2,number_of_points_chamfer).numpy()

    points_rm = []
    for i in range(number_of_points_chamfer):
        [xi1,xi2] = rm_ran_var[:,i]
        sqrt_xi1 = math.sqrt(xi1)
        rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
        rn_select = torch.randperm(3).numpy()
        u = rn[rn_select[0]]
        v = rn[rn_select[1]]
        w = rn[rn_select[2]]

        t_index = bisect_left(rm_triangle_cdf, rm_tri_samples[i])

        [v1,v2,v3] = np.array(rm.circulate_face(t_index,mode='v'))

        points_rm.append(u*rm.positions()[v1] + v*rm.positions()[v2] + w*rm.positions()[v3])
    points_rm = np.array(points_rm)


    # Create trees
    tree_gtm = KDTree(points_gtm)
    d_rm, _ = tree_gtm.query(points_rm, k=1, eps=0, p=2, distance_upper_bound=math.inf)

    # Create trees
    tree_rm = KDTree(points_rm)
    d_gtm, _ = tree_rm.query(points_gtm, k=1, eps=0, p=2, distance_upper_bound=math.inf)

    # ---------------------------------------------------------------------------- #
    # Chamfer distance
    # ---------------------------------------------------------------------------- #
    chamfer_distance = float((np.dot(d_rm,d_rm) + np.dot(d_gtm,d_gtm))/number_of_points_chamfer)

    # ---------------------------------------------------------------------------- #
    # Mesh accuracy
    # ---------------------------------------------------------------------------- #
    rm_tri_samples_accuracy = torch.rand(number_of_points_mesh_accuracy).numpy()
    
    rm_ran_var_accuracy = torch.rand(2,number_of_points_mesh_accuracy).numpy()

    points_rm_accuracy = []
    for i in range(number_of_points_mesh_accuracy):
        [xi1,xi2] = rm_ran_var_accuracy[:,i]
        sqrt_xi1 = math.sqrt(xi1)
        rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
        rn_select = torch.randperm(3).numpy()
        u = rn[rn_select[0]]
        v = rn[rn_select[1]]
        w = rn[rn_select[2]]

        t_index = bisect_left(rm_triangle_cdf, rm_tri_samples_accuracy[i])
    
        [v1,v2,v3] = np.array(rm.circulate_face(t_index,mode='v'))

        points_rm_accuracy.append(u*rm.positions()[v1] + v*rm.positions()[v2] + w*rm.positions()[v3])
    points_rm_accuracy = np.array(points_rm_accuracy)

    hmesh.MeshDistance(gtm)
    mesh_accuracy = abs(hmesh.MeshDistance(gtm).signed_distance(points_rm_accuracy)).reshape((len(points_rm_accuracy),1))

    return chamfer_distance, np.percentile(mesh_accuracy, 90)

# ------------------------------------------------------------------------------------------------------
# Compute the chamfer distance between two meshes using meshwise approach
#
# gtm:                      PyGEL3D mesh - ground truth mesh
# rm:                       PyGEL3D mesh - reconstructed mesh
# number_of_points_chamfer: int - Number of points used to compute chamfer distance
#
# Returns float
# ------------------------------------------------------------------------------------------------------
def calculate_chamfer_distance(gtm, rm, number_of_points_chamfer):
    # gtm - ground truth mesh
    # rm - reconstructed mesh

    # ---------------------------------------------------------------------------- #
    # Chamfer distance
    # ---------------------------------------------------------------------------- #
    # gtm arrays 
    gtm_triangle_areas = []
    for f in gtm.faces():
        gtm_triangle_areas.append(gtm.area(f))

    gtm_triangle_areas = [tri_area / sum(gtm_triangle_areas) for tri_area in gtm_triangle_areas]
    gtm_triangle_cdf = np.cumsum(gtm_triangle_areas)

    # rm arrays
    rm_triangle_areas = []
    for f in rm.faces():
        rm_triangle_areas.append(rm.area(f))

    rm_triangle_areas = [rm_tri_area / sum(rm_triangle_areas) for rm_tri_area in rm_triangle_areas]
    rm_triangle_cdf = np.cumsum(rm_triangle_areas)

    # Sample points on gtm
    gtm_tri_samples = torch.rand(number_of_points_chamfer).numpy()
    
    gtm_ran_var = torch.rand(2,number_of_points_chamfer).numpy()

    points_gtm = []
    for i in range(number_of_points_chamfer):
        [xi1,xi2] = gtm_ran_var[:,i]
        sqrt_xi1 = math.sqrt(xi1)
        rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
        rn_select = torch.randperm(3).numpy()
        u = rn[rn_select[0]]
        v = rn[rn_select[1]]
        w = rn[rn_select[2]]

        t_index = bisect_left(gtm_triangle_cdf, gtm_tri_samples[i])
   
        [v1,v2,v3] = np.array(gtm.circulate_face(t_index,mode='v'))

        points_gtm.append(u*gtm.positions()[v1] + v*gtm.positions()[v2] + w*gtm.positions()[v3])
    points_gtm = np.array(points_gtm)

    # Sample points on rm
    rm_tri_samples = torch.rand(number_of_points_chamfer).numpy()
    
    rm_ran_var = torch.rand(2,number_of_points_chamfer).numpy()

    points_rm = []
    for i in range(number_of_points_chamfer):
        [xi1,xi2] = rm_ran_var[:,i]
        sqrt_xi1 = math.sqrt(xi1)
        rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
        rn_select = torch.randperm(3).numpy()
        u = rn[rn_select[0]]
        v = rn[rn_select[1]]
        w = rn[rn_select[2]]

        t_index = bisect_left(rm_triangle_cdf, rm_tri_samples[i])

        [v1,v2,v3] = np.array(rm.circulate_face(t_index,mode='v'))

        points_rm.append(u*rm.positions()[v1] + v*rm.positions()[v2] + w*rm.positions()[v3])
    points_rm = np.array(points_rm)

    # Create trees
    tree_gtm = KDTree(points_gtm)
    d_rm, _ = tree_gtm.query(points_rm, k=1, eps=0, p=2, distance_upper_bound=math.inf)

    # Create trees
    tree_rm = KDTree(points_rm)
    d_gtm, _ = tree_rm.query(points_gtm, k=1, eps=0, p=2, distance_upper_bound=math.inf)

    # Chamfer distance
    chamfer_distance = float((np.dot(d_rm,d_rm) + np.dot(d_gtm,d_gtm))/number_of_points_chamfer)

    return chamfer_distance

# ------------------------------------------------------------------------------------------------------
# Compute the mesh completion measure between two meshes
#
# gtm:              PyGEL3D mesh - ground truth mesh
# rm:               PyGEL3D mesh - reconstructed mesh
#
# Returns float
# ------------------------------------------------------------------------------------------------------
def calculate_mesh_completion(gtm, rm, number_of_points, delta_distance, filepath, filename):
    
    gtm_sampled_points = os.path.join(filepath,filename + "_gtm_sampled_points_mesh_completion.npy")
    
    # ---------------------------------------------------------------------------- #
    # Sample points on ground truth mesh
    # ---------------------------------------------------------------------------- #
    if (not os.path.exists(gtm_sampled_points)):
        # gtm arrays 
        gtm_triangle_areas = []
        for f in gtm.faces():
            gtm_triangle_areas.append(gtm.area(f))

        gtm_triangle_areas = [tri_area / sum(gtm_triangle_areas) for tri_area in gtm_triangle_areas]
        gtm_triangle_cdf = np.cumsum(gtm_triangle_areas)

        # Sample points on gtm
        gtm_tri_samples = torch.rand(number_of_points).numpy()
        gtm_ran_var = torch.rand(2,number_of_points).numpy()

        points_gtm = []
        for i in range(number_of_points):
            [xi1,xi2] = gtm_ran_var[:,i]
            sqrt_xi1 = math.sqrt(xi1)
            rn = [1.0-sqrt_xi1, (1.0-xi2)*sqrt_xi1, xi2*sqrt_xi1]
            rn_select = torch.randperm(3).numpy()
            u = rn[rn_select[0]]
            v = rn[rn_select[1]]
            w = rn[rn_select[2]]

            t_index = bisect_left(gtm_triangle_cdf,  gtm_tri_samples[i])
          
            [v1,v2,v3] = np.array(gtm.circulate_face(t_index,mode='v'))

            points_gtm.append(u*gtm.positions()[v1] + v*gtm.positions()[v2] + w*gtm.positions()[v3])
        points_gtm = np.array(points_gtm)

        np.save(gtm_sampled_points,points_gtm)

    points_gtm = np.load(gtm_sampled_points)


    hmesh.MeshDistance(rm)
    point_distances = abs(hmesh.MeshDistance(rm).signed_distance(points_gtm)).reshape((len(points_gtm),1))

    mesh_completion_measure = float(sum(point_distances < delta_distance)/number_of_points)
    return mesh_completion_measure

def gelmesh_to_plotyly(m, c):
    xyz = np.array([ p for p in m.positions()])
    m_tri = hmesh.Manifold(m)
    hmesh.triangulate(m_tri)
    ijk = np.array([[ idx for idx in m_tri.circulate_face(f,'v')] for f in m_tri.faces()])
    mesh = go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],
            i=ijk[:,0],j=ijk[:,1],k=ijk[:,2],color=c,flatshading=True,opacity=0.50)
    return mesh

# ------------------------------------------------------------------------------------------------------
# Display two meshes at the same time
#
# m0:               PyGEL3D mesh - first mesh
# m1:               PyGEL3D mesh - second mesh
# ------------------------------------------------------------------------------------------------------  
def display_meshes(m0, m1):
    mesh0 = gelmesh_to_plotyly(m0, '#0000dd')
    mesh1 = gelmesh_to_plotyly(m1, '#dd0000')
    
    mesh_data = [mesh0, mesh1]

    fig = go.FigureWidget(data=mesh_data)
    fig.layout.scene.aspectmode="data"
    py.iplot(fig)

# ------------------------------------------------------------------------------------------------------
# Display two point sets at the same time
#
# P0:               np.array - Nx3 where N is number of points
# P1:               np.array - Nx3 where N is number of points
# ------------------------------------------------------------------------------------------------------  
def display_points(P0, P1):
    trace0 = go.Scatter3d(
        x = P0[:,0],
        y = P0[:,1],
        z = P0[:,2],
        mode = 'markers'
    )
    trace1 = go.Scatter3d(
        x = P1[:,0],
        y = P1[:,1],
        z = P1[:,2],
        mode = 'markers'
    )
    data = [trace0, trace1]
    fig = go.Figure(data=data)
    py.iplot(fig, filename='Annotation Points')

def display_mesh_and_points(m,P):
    trace = go.Scatter3d(
        x = P[:,0],
        y = P[:,1],
        z = P[:,2],
        mode = 'markers'
    )
    mesh = gelmesh_to_plotyly(m, '#dd0000')
    data = [trace, mesh]
    fig = go.FigureWidget(data=data)
    fig.layout.scene.aspectmode="data"
    py.iplot(fig)

# ------------------------------------------------------------------------------------------------------
# Function used to linearly interpolate between the values at two vertices
#
# Inputs:
# v1:               np.array - Position of vertex v1
# a:                float - GWN gradient length at vertex v1
# v2:               np.array - Position of vertex v2
# b:                float - GWN graident length at vertex v2
# threshold:        float - the threshold k in the article
# 
# Returns a float - the interpolated value
# ------------------------------------------------------------------------------------------------------  
def linear_interpolation_point(v1,a,v2,b,threshold):
    d = math.sqrt(np.dot(v2-v1,v2-v1))
    t = (d/(b-a))*(threshold-a) # distance along direction vector v
    v = (v2-v1)/d # direction vector
    return v1 + t*v

# ------------------------------------------------------------------------------------------------------
# Function which is used to remove the parts of the closed mesh, which should be holes according to 
# the GWN gradients 
#
# Inputs:
# m:                PYGEL3D mesh
# dv:               np.array - GWN gradient length computed at every vertex in the mesh
# threshold:        float - the threshold k in the article
# 
# Returns - PYGEL3D mesh
# ------------------------------------------------------------------------------------------------------  
def smooth_boundary_removal(m,dv,threshold):

    mesh = hmesh.Manifold(m)

    # dv = decision vector 
    intersection_info = []
    faces_to_be_removed = []
    for f in mesh.faces():
        [v0,v1,v2] = mesh.circulate_face(f,mode='v')
        [h0,h1,h2] = mesh.circulate_face(f,mode='h')
        
        # Case 1: All points are inside region
        if (np.sum(dv[np.array([v0,v1,v2])] <= threshold) == 3):
            #print("All points are inside")
            # remove face
            faces_to_be_removed.append(f)
        
        # Case 2: Two points are inside remove region
        elif (np.sum(dv[np.array([v0,v1,v2])] <= threshold) == 2):
            # v0 and v1 are outside
            if (dv[v0] <= threshold and dv[v1] <= threshold):
                p1 = linear_interpolation_point(mesh.positions()[v0],dv[v0],mesh.positions()[v2],dv[v2],threshold)
                p2 = linear_interpolation_point(mesh.positions()[v1],dv[v1],mesh.positions()[v2],dv[v2],threshold)
                intersection_info.append([h0,h2,p1,p2,f])
            
            # v0 and v2 are outside
            elif (dv[v0] <= threshold and dv[v2] <= threshold):
                p1 = linear_interpolation_point(mesh.positions()[v2],dv[v2],mesh.positions()[v1],dv[v1],threshold)
                p2 = linear_interpolation_point(mesh.positions()[v0],dv[v0],mesh.positions()[v1],dv[v1],threshold)
                intersection_info.append([h2,h1,p1,p2,f])
            
            # v1 and v2 are outside
            else:
                p1 = linear_interpolation_point(mesh.positions()[v1],dv[v1],mesh.positions()[v0],dv[v0],threshold) 
                p2 = linear_interpolation_point(mesh.positions()[v2],dv[v2],mesh.positions()[v0],dv[v0],threshold)
                intersection_info.append([h1,h0,p1,p2,f])
        
        # Case 3: One point is inside region
        elif (np.sum(dv[np.array([v0,v1,v2])] <= threshold) == 1):
            # v0 is outside
            if (dv[v0] <= threshold):
                p1 = linear_interpolation_point(mesh.positions()[v0],dv[v0],mesh.positions()[v2],dv[v2],threshold) 
                p2 = linear_interpolation_point(mesh.positions()[v0],dv[v0],mesh.positions()[v1],dv[v1],threshold)
                intersection_info.append([h0,h1,p1,p2,f])
            
            # v1 is outside
            elif (dv[v1] <= threshold):
                p1 = linear_interpolation_point(mesh.positions()[v1],dv[v1],mesh.positions()[v0],dv[v0],threshold)
                p2 = linear_interpolation_point(mesh.positions()[v1],dv[v1],mesh.positions()[v2],dv[v2],threshold) 
                intersection_info.append([h1,h2,p1,p2,f])
            
            # v2 is otuside
            else:
                p1 = linear_interpolation_point(mesh.positions()[v2],dv[v2],mesh.positions()[v1],dv[v1],threshold)
                p2 = linear_interpolation_point(mesh.positions()[v2],dv[v2],mesh.positions()[v0],dv[v0],threshold) 
                intersection_info.append([h2,h0,p1,p2,f])
    
    # Split faces
    for i in range(len(intersection_info)):
        [h1,h2,p1,p2,f] = intersection_info[i]
        v1 = mesh.split_edge(h1)
        v2 = mesh.split_edge(h2)
        mesh.positions()[v1] = p1
        mesh.positions()[v2] = p2
        f_for_removal = mesh.split_face_by_edge(f,v2,v1)
        faces_to_be_removed.append(f_for_removal)
    
    # remove faces
    for j in range(len(faces_to_be_removed)):
        mesh.remove_face(faces_to_be_removed[j])        
    
    hmesh.stitch(mesh)
    mesh.cleanup()
    hmesh.triangulate(mesh, clip_ear=True)
    if (not check_if_triangle_mesh(mesh)):
        hmesh.triangulate(mesh, clip_ear=False)

    return mesh