import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .build import MODELS
from utils.logger import *
import random
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.transformer import Group, Encoder, TransformerEncoder, TransformerDecoder  

from models.RandomCrop import pointRandomCrop
from timm.models.layers import trunc_normal_

from models.transformer import CrossAttention
from models.pos import get_pos_embed


class Encoder_Blocks(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(  
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, neighborhood, center, noaug=False):
        """
        neighborhood (B, G, M, 3)
        center (B, G, 3)
        """
        group_input_tokens = self.encoder(neighborhood)  # B G N 3 -> B G C   
        pos = self.pos_embed(center)        
        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)   
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        x = torch.cat((cls_token, group_input_tokens), dim=1)     
        pos = torch.cat((cls_pos, pos), dim=1)
        
        x = self.blocks(x, pos)
        x = self.norm(x)
        return x[:, :1],x[:, 1:]



@MODELS.register_module()
class Point_PQAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point-PQAE] ', logger='Point-PQAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.PQAE_encoder = Encoder_Blocks(config)          # NOTE   
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]

        print_log(f'[Point-PQAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point-PQAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        self.crop_rate_min = config.min_crop_rate
        
        self.PQ_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        self.PQattn_contrast = CrossAttention(self.trans_dim, num_heads=self.decoder_num_heads, qkv_bias=False, 
                                        qk_scale=None, attn_drop=0., proj_drop=0.)  # position query
        self.PQ_linear = nn.Linear(self.trans_dim, 3 * self.group_size)
        
        self.loss_type = config.loss_type
        if config.loss_type in ['cdl1', 'cdl2']:
            self.build_loss_func(config.loss_type)
        elif config.loss_type == 'cos':
            pass 
        else:
            raise NotImplementedError
        
        self.apply(self._init_weights)      
        
    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, pts, vis=False, **kwargs):
        losses = {}
        pts2_minus_pts1, pts1, pts2, angle1, angle2 = pointRandomCrop(pts, crop_rate_min=self.crop_rate_min) # pts: bs, M, 3
        neighborhood1, center1 = self.group_divider(pts1)       # B, G, M, 3 & B, G, 3
        neighborhood2, center2 = self.group_divider(pts2) 
        
        cls_token1, encoded_pts1 = self.PQAE_encoder(neighborhood1, center1)     
        cls_token2, encoded_pts2 = self.PQAE_encoder(neighborhood2, center2)
        batch_size, _, C = encoded_pts1.shape   # B, G, C
        
        # positional query process
        B, G, C = encoded_pts2.shape
        pts2_minus_pts1 = pts2_minus_pts1.unsqueeze(1).expand(-1, G, -1)   
        pts1_minus_pts2 = -pts2_minus_pts1
        relative1 = torch.cat([center1, pts2_minus_pts1], dim=-1)     
        relative2 = torch.cat([center2, pts1_minus_pts2], dim=-1)    

        pos_embed1 = get_pos_embed(self.trans_dim, relative1)
        pos_embed2 = get_pos_embed(self.trans_dim, relative2)
        # pos_embed1 = torch.randn([128, 64, 384]).cuda()
        # pos_embed2 = torch.randn([128, 64, 384]).cuda()   # For ablation study -> None positional embedding
        # For ablation APE: crop but no norm., keeping the two pts in the same coordinate system

        predict11 = self.PQattn_contrast(pos_embed1, encoded_pts2)     
        predict22 = self.PQattn_contrast(pos_embed2, encoded_pts1)
        # predict11 = encoded_pts2 
        # predict22 = encoded_pts1

        # Decoder
        predict11 = self.PQ_decoder(predict11)
        predict22 = self.PQ_decoder(predict22)        
            
        if 'cos' in self.loss_type:    
            predict1 = F.normalize(self.PQ_linear(predict11), dim=-1)      
            predict2 = F.normalize(self.PQ_linear(predict22), dim=-1)
        else:       # cdl2,
            predict1 = self.PQ_linear(predict11)
            predict2 = self.PQ_linear(predict22)
            
        target1 = neighborhood1.view(B, G, -1)    # B, G, 3 * M
        target2 = neighborhood2.view(B, G, -1)    # B, G, 3 * M
        
        if 'cos' in self.loss_type:  
            target1 = F.normalize(target1, dim=-1)
            target2 = F.normalize(target2, dim=-1)
            losses['pq_loss1'] = -(predict1 * target1.detach()).sum(dim=-1).mean() 
            losses['pq_loss2'] = -(predict2 * target2.detach()).sum(dim=-1).mean()
        elif self.loss_type in ['cdl1', 'cdl2']:  
            losses['pq_loss1'] = self.loss_func(predict1.reshape(B * G, -1, 3), target1.reshape(B * G, -1, 3).detach())
            losses['pq_loss2'] = self.loss_func(predict2.reshape(B * G, -1, 3), target2.reshape(B * G, -1, 3).detach())
        else:
            raise NotImplementedError
        if vis:
            # return neighborhood1, predict1, neighborhood2, predict2
            # neighborhood B, 64, 32, 3
            # center B, 64, 3
            neighborhood1 = neighborhood1 + center1.unsqueeze(2)
            neighborhood2 = neighborhood2 + center2.unsqueeze(2)
            predict1 = predict1.reshape(B, -1 ,self.group_size,3) + center1.unsqueeze(2)
            predict2 = predict2.reshape(B, -1 ,self.group_size,3) + center2.unsqueeze(2)        
            return neighborhood1, predict1, neighborhood2, predict2, center1, center2
        
        return losses