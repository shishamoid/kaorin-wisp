# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math
import copy

from wisp.ops.spc import sample_spc
from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.geometric import sample_unif_sphere

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.accelstructs import OctreeAS
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import *

import kaolin.ops.spc as spc_ops

class NeuralRadianceField(BaseNeuralField):
    """Model for encoding radiance fields (density and plenoptic color)
    """
    def init_embedder(self):
        """Creates positional embedding functions for the position and view direction.
        """
        self.pos_embedder, self.pos_embed_dim = get_positional_embedder(self.pos_multires, 
                                                                       self.embedder_type == "positional")
        self.view_embedder, self.view_embed_dim = get_positional_embedder(self.view_multires, 
                                                                         self.embedder_type == "positional")
        log.info(f"Position Embed Dim: {self.pos_embed_dim}")
        log.info(f"View Embed Dim: {self.view_embed_dim}")
        print(f"Position Embed Dim: {self.pos_embed_dim}")
        print(f"View Embed Dim: {self.view_embed_dim}")

    def init_decoder(self):
        """Initializes the decoder object. 
        """
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim

        self.input_dim = self.effective_feature_dim + self.view_embed_dim

        if self.position_input:
            self.input_dim += self.pos_embed_dim
            
        #追加
        print("###############")
        print(self.noise_dim)
        #assert False
        print("input_dim",self.input_dim) #59
        print("self.activation_type",self.activation_type)
        print("get_activation_class",get_activation_class(self.activation_type))
        print("get_layer_class",get_layer_class(self.layer_type))
        print("###############")
        self.decoder = BasicDecoder(self.input_dim, 4, get_activation_class(self.activation_type), True,
                                    layer=get_layer_class(self.layer_type), num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim, noise_dim=self.noise_dim,noise_size = self.noise_size,skip=[])

    def init_grid(self):
        """Initialize the grid object.
        """
        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif self.grid_type == "HashGrid":
            grid_class = HashGrid
        else:
            raise NotImplementedError

        self.grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.grid is not None:
            
            if self.grid_type == "HashGrid":
                # TODO(ttakikawa): Expose these parameters. 
                # This is still an experimental feature for the most part. It does work however.
                density_decay = 0.6
                min_density = ((0.01 * 512)/np.sqrt(3))

                self.grid.occupancy = self.grid.occupancy.cuda()
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points.cuda()
                #idx = torch.randperm(points.shape[0]) # [:N] to subsample
                res = 2.0**self.grid.blas_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
                with torch.no_grad():
                    density = self.forward(coords=samples[:,None], ray_d=sample_views, channels="density")
                self.grid.occupancy = torch.stack([density[:, 0, 0], self.grid.occupancy], -1).max(dim=-1)[0]

                mask = self.grid.occupancy > min_density
                
                #print(density.mean())
                #print(density.max())
                #print(mask.sum())
                #print(self.grid.occupancy.max())

                _points = points[mask]
                octree = spc_ops.unbatched_points_to_octree(_points, self.grid.blas_level, sorted=True)
                self.grid.blas.init(octree)
            else:
                raise NotImplementedError

    def get_nef_type(self):
        """Returns a text keyword of the neural field type.

        Returns:
            (str): The key type
        """
        return 'nerf'

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ["density", "rgb"])

    def rgba(self, coords, ray_d, pidx=None, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, num_samples, 3] 
                - Density tensor of shape [batch, num_samples, 1]
        """
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, num_samples, _ = coords.shape
        timer.check("rf_rgba_preprocess")
        
        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        timer.check("rf_rgba_interpolate")

        # Optionally concat the positions to the embedding, and also concatenate embedded view directions.

        if self.position_input:
            fdir = torch.cat([feats,
                self.pos_embedder(coords.reshape(-1, 3)),
                self.view_embedder(-ray_d)[:,None].repeat(1, num_samples, 1).view(-1, self.view_embed_dim)], dim=-1)
        else: 
            fdir = torch.cat([feats,
                self.view_embedder(-ray_d)[:,None].repeat(1, num_samples, 1).view(-1, self.view_embed_dim)], dim=-1)

        #乱数追加部分
        noize_dim = self.noise_dim
        noise_size = self.noise_size
        
        torch.manual_seed(0)
        arr = np.random.randn(int(fdir.size(dim=0)),noize_dim)*noise_size
        tensorx = torch.from_numpy(arr.astype(np.float32)).clone()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensorx = tensorx.to(device)
        fdir = torch.cat([tensorx,fdir],dim=1) #乱数追加
        
        timer.check("rf_rgba_embed_cat")

        # Decode high-dimensional vectors to RGBA.
        rgba = self.decoder(fdir)
        timer.check("rf_rgba_decode")

        # Colors are values [0, 1] floats
        colors = torch.sigmoid(rgba[...,:3]).reshape(batch, num_samples, -1)

        # Density is [particles / meter], so need to be multiplied by distance
        density = torch.relu(rgba[...,3:4]).reshape(batch, num_samples, -1)
        timer.check("rf_rgba_activation")
        
        return dict(rgb=colors, density=density)
