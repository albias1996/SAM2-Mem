import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.xmem_modules.group_modules import *
from sam2.xmem_modules.model.cbam import CBAM
from sam2.xmem_modules.model import resnet

################################## first modules are for the key projection

class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        #nn.init.orthogonal_(self.key_proj.weight.data)
        #nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        ###shrinkage outputs a value per pixel --> squared to ensure it is positive and +1 to ensure minimum shrinkage is 1 (neutral confidence)
        ###so s controls how much each memory key trusts itself — higher shrinkage → more local influence
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None 
        ###selection term outputs a value per channel per pixel with range [0,1]
        ###for this pixel, put more attention weight on these channels.” It gates the per-channel contribution during similarity computation
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection

class XMemFeatureProjector(nn.Module):
    def __init__(self, fused_feature_channels, key_dim):
        super().__init__()
        self.key_proj = KeyProjection(fused_feature_channels, key_dim)

    def forward(self, fused_feature, need_sk=True, need_ek=True):
        key, shrinkage, selection = self.key_proj(fused_feature, need_sk, need_ek)
        
        return key, shrinkage, selection

###################################

######################################Lightweight modules for the GRU_Update

class UpsampleBlockSep(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        # 4D separable conv for main features
        self.skip_conv = SepMainConv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)

        self.distributor = MainToGroupDistributor(method='add')

        # group-aware separable block for grouped tensor g
        self.out_conv = GroupResBlockSep(g_up_dim, g_out_dim)

        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)                 # skip_f: [B, g_up_dim, ...] (4D)
        g = upsample_groups(up_g, ratio=self.scale_factor)  # g: [B, O, ..., H, W] (5D)
        g = self.distributor(skip_f, g)                 # fuse main → grouped
        g = self.out_conv(g)                            # group-aware convs
        return g

class FeatureFusionBlockSep(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()
        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlockSep(x_in_dim + g_in_dim, g_mid_dim)  # keep g_mid_dim=512
        self.attention = CBAM(g_mid_dim)                                # unchanged
        self.block2 = GroupResBlockSep(g_mid_dim, g_out_dim)            # keep g_out_dim=512

    def forward(self, x, g):
        B, O = g.shape[:2]
        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(0,1)).view(B, O, -1, g.size(-2), g.size(-1))
        g = self.block2(g + r)
        return g

################################### 

################################# the following modules are for the GRU

class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g

class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)
        # self.transform = SepGConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        if isinstance(self.transform, SepGConv2D):
            nn.init.xavier_normal_(self.transform.depthwise.weight)
            nn.init.xavier_normal_(self.transform.pointwise.weight)
        else:
            nn.init.xavier_normal_(self.transform.weight)

        # nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class GRU_Update(nn.Module):
    def __init__(self, value_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        ###those are the 3 blocks I need to change input channels (at least) to fit the input of the GRU (note: 1024+512+256=1792)
        self.fuser = FeatureFusionBlock(1024, value_dim + hidden_dim, 512, 512)
        # self.fuser = FeatureFusionBlockSep(1024, value_dim + hidden_dim, 512, 512)


        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)   # 1/8 -> 1/4
        # upsample path (same dims as before)
        # self.up_16_8 = UpsampleBlockSep(512, 512, 256)
        # self.up_8_4  = UpsampleBlockSep(256, 256, 256)

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)
        
        #also need to change the input channels of hidden_update 
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256 + 1], 256, hidden_dim)
        else:
            self.hidden_update = None 

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        """
        Args:
            f16, f8, f4: multi-scale features
            hidden_state: previous GRU state, shape [1, num_objects, CH, H/16, W/16]
            memory_readout: readout from memory, shape [1, num_objects, CV, H/16, W/16]

        Returns:
            updated hidden_state, g16
        """
        batch_size, num_objects = memory_readout.shape[:2]
        
        # Step 1: Fuse memory and hidden
        fused = torch.cat([memory_readout, hidden_state], dim=2) #torch.Size([1, 2, 576, 64, 64])
        if self.hidden_update is not None:
            g16 = self.fuser(f16, fused)  #torch.Size([1, 2, 512, 64, 64])
        else:
            g16 = self.fuser(f16, memory_readout)

        # Step 2: Upsample
        g8 = self.up_16_8(f8, g16) #torch.Size([1, 2, 256, 128, 128])
        g4 = self.up_8_4(f4, g8)  #torch.Size([1, 2, 256, 256, 256])

        # Step 3: Predict logits
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))
        logits = logits.view(batch_size, num_objects, 1, *logits.shape[-2:])

        # Step 4: GRU update
        if h_out and self.hidden_update is not None:
            g4_cat = torch.cat([g4, logits], dim=2)
            updated_hidden = self.hidden_update([g16, g8, g4_cat], hidden_state)
        else:
            updated_hidden = None

        return updated_hidden, g16

################################### the next modules are for the value encoder 

class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim, light_version_value_encoder=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        if not light_version_value_encoder:
            self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)
        else:
            self.transform = SepGConv2D(g_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)

        if isinstance(self.transform, SepGConv2D):
            nn.init.xavier_normal_(self.transform.depthwise.weight)
            nn.init.xavier_normal_(self.transform.pointwise.weight)
        else:
            nn.init.xavier_normal_(self.transform.weight)

        # nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h

class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False, light_version_value_encoder=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        if not light_version_value_encoder:
            self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        else:
            self.fuser = FeatureFusionBlockSep(1024, 256, value_dim, value_dim)

        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim, light_version_value_encoder)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g = self.layer1(g) # 1/4
        g = self.layer2(g) # 1/8
        g = self.layer3(g) # 1/16

        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.fuser(image_feat_f16, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h



