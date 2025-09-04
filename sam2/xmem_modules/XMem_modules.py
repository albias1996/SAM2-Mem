
import torch
import torch.nn as nn
from sam2.xmem_modules.modules import XMemFeatureProjector, GRU_Update, ValueEncoder
from sam2.xmem_modules.memory_util import *

class XMem(nn.Module):
    def __init__(self, memory_cfg, hidden_dim_sam2=256, sam_prompt_embed_dim=256):
        """
        Only define all weight‐bearing submodules here. Do NOT implement any time‐step loops.
        """
        super().__init__()
        self.memory_cfg = memory_cfg
        self.hidden_dim_sam2 = hidden_dim_sam2
        self.sam_prompt_embed_dim = sam_prompt_embed_dim
        self.value_dim = memory_cfg["value_dim"]
        self.xmem_projector = XMemFeatureProjector(fused_feature_channels=1024, key_dim=memory_cfg["key_dim"])

        # Feature projection (SAM2 FPN to XMem input)
        self.f16_to_xmem = nn.Conv2d(hidden_dim_sam2, memory_cfg["out_channels_f16"], 1, bias=False)
        self.f8_to_xmem = nn.Conv2d(int(hidden_dim_sam2/4), memory_cfg["out_channels_f8"], 1, bias=False)
        self.f4_to_xmem = nn.Conv2d(int(hidden_dim_sam2/8), memory_cfg["out_channels_f4"], 1, bias=False)

        # GRU for hidden state update
        self.sensory_gru = GRU_Update(memory_cfg["value_dim"], memory_cfg["hidden_dim"])

        # Memory value encoder
        self.xmem_value_encoder = ValueEncoder(
            value_dim=memory_cfg["value_dim"],
            hidden_dim=memory_cfg["hidden_dim"],
            single_object=memory_cfg["single_object"],
            light_version_value_encoder=memory_cfg["light_version_value_encoder"]
        )

        # Fusion conv to align hidden + value output with SAM2 features
        self.adapter = nn.Conv2d(memory_cfg["value_dim"], sam_prompt_embed_dim, kernel_size=1)
        
        #self.deep_update_prob = cfg.get("deep_update_prob", 0.5)

        # self.occlusion_embedding = nn.Parameter(torch.randn(memory_cfg["value_dim"]))

        # self.obj_ptr_to_mem_dim = nn.Conv2d(hidden_dim_sam2, memory_cfg["value_dim"], kernel_size=1)

    def key_projection(self, frame, f16, need_sk=True, need_ek=True):
        """
        Project input features to key space.
        """
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w --> t is the number of frames 
            need_reshape = True
            b, t = f16.shape[:2]
            f16 = f16.flatten(start_dim=0, end_dim=1) ### convert: [B, T, C, H, W] → [B*T, C, H, W]
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        f16 = self.f16_to_xmem(f16)
        key, shrinkage, selection = self.xmem_projector(f16, need_sk=need_sk, need_ek=need_ek)

        if need_reshape:
            # B*C*T*H*W
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

        return key, shrinkage, selection

    def adapter_layer(self, frame_embedding):
        """
        Apply adapter conv to fit frame embedding dimensions with sam2 decoder input ones.
        """

        # Apply adapter conv
        #frame_embedding before shape is torch.Size([B, O, 512, 64, 64]) where O is the number of objects and B is the batch size
        frame_embedding_adapted = self.adapter(frame_embedding)

        return frame_embedding_adapted

    
    def update_hidden_state(self, f16, f8, f4, hidden_state, memory_readout, h_out):
        """
        Update the hidden state using GRU.
        """
        # Project features to XMem input space
        f16 = self.f16_to_xmem(f16)
        f8 = self.f8_to_xmem(f8)
        f4 = self.f4_to_xmem(f4)

        # Update hidden state using GRU
        updated_hidden_state, frame_embedding = self.sensory_gru(f16, f8, f4, hidden_state, memory_readout, h_out)

        return updated_hidden_state, frame_embedding


    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True):
        """
        Encode the value for the current frame.
        """
        #1) masks[:, i] is mask for object i, others[:, i] is sum of all masks ≠ i --> purpose : Give each object a view of “everything else”
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)
        ##2) compute the memory value v and hidden state h_t 
        image_feat_f16 = self.f16_to_xmem(image_feat_f16)
        g16, h16 = self.xmem_value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g16, h16



    # Used in training only. 
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key, 
                    memory_shrinkage, memory_value):                  
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    
    # def adapter_obj_pointer(self, obj_ptr, memory_readout):
    #     """
    #     Adapt object pointer to memory readout dimensions.
    #     """
    #     B, O, CV, H, W = memory_readout.shape

    #     # Project object pointer to memory dimension
    #     e_token = obj_ptr.unsqueeze(-1).unsqueeze(-1) # → [B*O, mem_dim, 1, 1]

    #     e_spatial = e_token.expand(-1, -1, H, W)  # → [B*O, mem_dim, H, W]

    #     e_final = self.obj_ptr_to_mem_dim(e_spatial) # → [B*O, value_dim, H, W]

    #     #reshap dim0 and dim 1 to B and O
    #     e_final = e_final.view(B, O, self.value_dim, H, W)

    #     #add memory readout
    #     memory_readout = e_final + memory_readout
        
    #     return memory_readout


    # def occlusion_score_embedding(self, values, is_occluded):
    #     """
    #     Compute occlusion score embedding.
    #     N: number of objects
    #     values: B * N * CV * H * W
    #     is_occluded: B * N * 1 * 1 * 1
    #     """
    #     # Create occlusion embedding
    #     occlusion_embedding = self.occlusion_embedding.view(1, 1, self.value_dim, 1, 1).expand(1, values.shape[1], self.value_dim, 1, 1)
        
    #     # Apply occlusion embedding based on is_occluded
    #     occlusion_score_embedding = occlusion_embedding * is_occluded

    #     values = values + occlusion_score_embedding

    #     return values

