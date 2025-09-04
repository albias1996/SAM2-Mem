# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

import numpy as np
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_base_no_mem import SAM2BaseXMem
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint
from sam2.xmem_modules.XMem_modules import XMem
from sam2.xmem_modules.model.aggregate import aggregate

class SAM2XMemTrain(SAM2BaseXMem):
    """
    A training‐time version of SAM2 + XMem. In `forward(batch)`, it should
    combine:
    1) the SAM2 “backbone + prompt"  (exactly as in SAM2Train.forward),
    2) the XMem modules (project fused features into memory keys, run the GRU, etc.),
    3) the SAM2 decoder (prompt‐decoder) to produce masks,
    """
    def __init__(
        self, 
        image_encoder,
        #memory_attention=None,
        #memory_encoder=None, 
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        xmem_ckpt_path=None,
        freeze_image_encoder=False,
        freeze_sam_mask_decoder=False,
        freeze_sam_prompt_encoder=False,
        freeze_xmem_feature_projector=False,
        freeze_value_encoder=False,
        freeze_sensory_gru=False,

        **kwargs,
    ):
        self.memory_cfg = kwargs.pop("memory_cfg", None)
        
        super().__init__(image_encoder, **kwargs)

        # Now build whatever XMem submodules you want (projectors, GRU, value‐encoder, etc.).
        self.xmem = XMem(memory_cfg=self.memory_cfg) 

        # xmem_checkpoint_path = "/scratch_net/thor_second/master_thesis/XMem/saves/XMem.pth"
        # xmem_checkpoint_path = None

        # 2) Load XMem weights (only the matching parts)
        if xmem_ckpt_path is not None:
            self._load_xmem_weights(xmem_ckpt_path)

        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        # add a variable that contain the obj_pointer 
        self.actual_obj_pointer = None

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_sam_prompt_encoder:
            for p in self.sam_prompt_encoder.parameters():
                p.requires_grad = False

        if freeze_sam_mask_decoder:
            for p in self.sam_mask_decoder.parameters():
                p.requires_grad = False

        if freeze_value_encoder:
            for p in self.xmem.xmem_value_encoder.parameters():
                p.requires_grad = False

        if freeze_xmem_feature_projector:
            for p in self.xmem.xmem_projector.parameters():
                p.requires_grad = False

        if freeze_sensory_gru:
            for p in self.xmem.sensory_gru.parameters():
                p.requires_grad = False
    
    def _load_xmem_weights(self, xmem_ckpt_path):
        xmem_sd = torch.load(xmem_ckpt_path, map_location="cpu")

        def load_sub(name, module, prefix):
            subdict = {
                k[len(prefix) + 1:]: v
                for k, v in xmem_sd.items()
                if k.startswith(prefix + ".")
            }
            missing, unexpected = module.load_state_dict(subdict, strict=False)
            if missing:
                logging.warning(f"[XMem:{name}] Missing keys:\n{missing}")
            if unexpected:
                logging.warning(f"[XMem:{name}] Unexpected keys:\n{unexpected}")

            logging.info(f"[XMem:{name}] Loaded {len(subdict)}/{len(module.state_dict())} keys")

        load_sub("value_encoder", self.xmem.xmem_value_encoder, prefix="value_encoder")
        load_sub("key_proj", self.xmem.xmem_projector.key_proj, prefix="key_proj")
        load_sub("decoder", self.xmem.sensory_gru, prefix="decoder")

        logging.info("✅ Loaded XMem weights")


    def forward(self, input: BatchedVideoDatapoint):
        """
        Forward pass for training. This combines the SAM2 backbone, XMem modules,
        and SAM2 decoder to produce masks.
        """
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch) #shape of input.flat_img_batch is torch.Size([8, 3, 1024, 1024]) note that T=8 (and can be change from the yaml file)
            #backbone_out.keys() = dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
            #len(backbone_out['backbone_fpn'])=3 with [0] of shape torch.Size([2, 32, 256, 256]), [1] of shape torch.Size([2, 64, 128, 128]), [2] of shape torch.Size([2, 256, 64, 64])	
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        
        #adapt format data to fit into memory manager
        data_xmem_style = self.make_xmem_input_from_sam2_batch(input, C=self.memory_cfg["max_num_objects"])

        selector = data_xmem_style["selector"]  # [B, C_slots, 1, 1]
        backbone_out = self.prepare_prompt_inputs(backbone_out, input, selector)
        #backbone_out.keys() = dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn', 'gt_masks_per_frame', 'num_frames', 
        #'use_pt_input', 'init_cond_frames', 'frames_not_in_init_cond', 'mask_inputs_per_frame', 'point_inputs_per_frame', 'frames_to_add_correction_pt'])

        ##continue the forward pass
        previous_stages_out = self.forward_tracking_xmem_style(backbone_out, input, data_xmem_style)
        #len(previous_stages_out)=/ (here 2) where both elements are dict 
        # previous_stages_out[i].keys() = dict_keys(['point_inputs', 'mask_inputs', 'multistep_pred_masks', 'multistep_pred_masks_high_res', 'multistep_pred_multimasks', 
        #'multistep_pred_multimasks_high_res', 'multistep_pred_ious', 'multistep_point_inputs', 'multistep_object_score_logits', 'pred_masks', 'pred_masks_high_res', 'maskmem_features', 'maskmem_pos_enc'])
        return previous_stages_out

    def prepare_prompt_inputs(self, backbone_out, input,  selector, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }

        # test_gt_masks_per_frame = {
        #     stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, masks in enumerate(input.masks)
        # }


        # gt_masks_per_frame = {
        #     stage_id: torch.cat(
        #         [
        #             masks.unsqueeze(1),  # shape [B, 1, H_im, W_im]
        #             torch.zeros(
        #                 C_slots - masks.shape[0], 1, masks.shape[1], masks.shape[2],
        #                 device=masks.device,
        #                 dtype=masks.dtype
        #             )
        #         ],
        #         dim=0
        #     ) if masks.shape[0] < C_slots else masks.unsqueeze(1)
        #     for stage_id, masks in enumerate(input.masks)
        # }

        gt_masks_per_frame = {}

        # selector: [B, C_slots] of 0/1 telling you which slots are active per video
        B, C_slots, _, _ = selector.shape
        # how many real objects per video
        num_real = selector.sum(dim=1).long()  # [B]

        for t, masks_t in enumerate(input.masks):
            # masks_t: [O_t, H, W]
            O_t, H, W = masks_t.shape

            # 1) allocate [B, C_slots, H, W]
            fixed = masks_t.new_zeros((B, C_slots, H, W), dtype=torch.bool)

            # 2) figure out which global‐slot index belongs to which video
            #    input.obj_to_frame_idx[t] is assumed [O_t, 2] giving (frame_idx, video_idx)
            video_idx = input.obj_to_frame_idx[t][:, 1]  # [O_t]

            # 3) for each video b, copy its first num_real[b] masks into fixed[b]
            for b in range(B):
                # indices in the global slot dimension that belong to video b
                all_slots_b = (video_idx == b).nonzero(as_tuple=False).squeeze(1)  # [O_b]
                n = num_real[b].item()
                for j, slot in enumerate(all_slots_b[:n]):
                    fixed[b, j] = masks_t[slot]

            # 4) if your downstream wants [B*C_slots, 1, H, W], flatten & unsqueeze:
            out = fixed.flatten(0, 1).unsqueeze(1)  # [B*C_slots, 1, H, W]

            gt_masks_per_frame[t] = out

        # gt_masks_per_frame = input.masks.unsqueeze(2) # [T,B,1,H_im,W_im] keep everything in tensor form
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames
        
        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(
                        gt_masks_per_frame[t],
                    )
                else:
                    # (here we only sample **one initial point** on initial conditioning frames from the
                    # ground-truth mask; we may sample more correction points on the fly)
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking_xmem_style(self, backbone_out: dict, input: BatchedVideoDatapoint, data_xmem_style: dict, return_dict=False):
        
        #1. prepare the features and set up the non/initial conditioning frames
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)
        
        # Starting the stage loop
        num_frames = backbone_out["num_frames"] #8
        init_cond_frames = backbone_out["init_cond_frames"] #[0] --> always the first frame (with GT masks) we will not do anything else 
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"] #[] <-- no frame to add correction points
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }

        # 2. Unpack the XMem‐style inputs (“rgb”, “first_frame_gt”, etc.)
        rgbs           = data_xmem_style["rgb"]           # [B, T, 3, H, W]
        first_frame_gt = data_xmem_style["first_frame_gt"] # [B, 1, C_slots, H, W]
        selector       = data_xmem_style["selector"]       # [B, C_slots, 1, 1]
        num_filled_objects = data_xmem_style["num_filled_objects"]    # length‐B tensor --> only the real objects
        num_objects    = first_frame_gt.shape[2] #this count also the "phantom" objects
        real_obj = num_filled_objects.sum().item() 
        B, T, _, H_img, W_img = rgbs.shape


        (f4_raw, f8_raw, f16_raw) = vision_feats[-3:]
        f4 = self.reshape_feat(f4_raw, feat_sizes[-3], input) #(torch.Size([B, 8, 32, 256, 256])
        f8 = self.reshape_feat(f8_raw, feat_sizes[-2], input) #torch.Size([B, 8, 64, 128, 128])
        f16 = self.reshape_feat(f16_raw, feat_sizes[-1], input) #torch.Size([B, 8, 256, 64, 64])

        (pos4_raw, pos8_raw, pos16_raw) = vision_pos_embeds[-3:]
        pos16 = self.reshape_feat(vision_pos_embeds[-1], feat_sizes[-1], input) #torch.Size([B, 8, 256, 64, 64])

        # 3. extract key, shrinkage, selection for all frames and initialize the hidden state
        key, shrinkage, selection = self.xmem.key_projection(input.img_batch, f16, need_sk=True, need_ek=True)
        #key.shape = torch.Size([B, 64, 8, 64, 64])
        #shrinkage.shape = torch.Size([B, 1, 8, 64, 64])
        #selection.shape = torch.Size([B, 64, 8, 64, 64])
        
        # 4. initialize the hidden state & encode all the objects for frame 0
        filler_one = torch.zeros(1, dtype=torch.int64)
        hidden = torch.zeros((rgbs.shape[0], self.memory_cfg["max_num_objects"], self.memory_cfg['hidden_dim'], *key.shape[-2:]), device=self.device)
        ##shape of tensors before feeding into xmem.encode_value
        #rgbs[:,0].shape=torch.Size([B, 3, 1024, 1024])
        #f16[:,0].shape=torch.Size([B, 256, 64, 64])
        #hidden.shape=torch.Size([B, max_obj, 64, 64, 64])
        #first_frame_gt[:,0].shape=.Size([B, max_obj, 1024, 1024]) 

        v16, hidden = self.xmem.encode_value(rgbs[:,0], f16[:,0], hidden, first_frame_gt[:,0])
        values = v16.unsqueeze(3) # add the time dimension --> torch.Size([B, max_obj, 512, 1, 64, 64])
        # 5. Loop over all frames in the processing order
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            img_ids = input.flat_obj_to_img_idx[stage_id]

            C_slots = self.memory_cfg["max_num_objects"]    # e.g. 3
            filled = num_filled_objects.tolist()            # e.g. [3,3,3,2]

            # create a map from video b to its list of img_ids:
            # note: img_ids is length sum(filled), entries in range [0..B-1]
            # we assume it's grouped by video already (as Sam2 does)
            groups = []
            start = 0
            for b, n in enumerate(filled):
                group = img_ids[start : start + n]          # Tensor of length n
                start += n
                groups.append(group)

            # now pad each group to length C_slots by repeating group[0]
            padded = []
            for group in groups:
                if len(group) < C_slots:
                    pad_count = C_slots - len(group)
                    # repeat first index pad_count times
                    pad_idxs = group[:1].expand(pad_count)
                    group = torch.cat([group, pad_idxs], dim=0)
                padded.append(group)

            # finally concatenate into a single long index vector
            img_ids_padded = torch.cat(padded, dim=0) 

            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats_inputs = [x[:, img_ids] for x in vision_feats] #[0] --> torch.Size([65536, num_objects, 32]), [1] --> torch.Size([16384, num_objects, 64]), [2] --> torch.Size([4096, num_objects, 256]) 
                # current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]

                current_vision_feats = [x[:, img_ids_padded] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids_padded] for x in vision_pos_embeds]

            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (_, current_vision_feats, current_vision_pos_embeds, feat_sizes) = self._prepare_backbone_features_per_frame(input.flat_img_batch, img_ids)

            t = stage_id
            is_init = (t in init_cond_frames)
            is_last = (t == num_frames - 1)

            # 1. skip the initial seed frame (we already decoded it and seeded memory)
            if (stage_id in init_cond_frames) and (backbone_out["mask_inputs_per_frame"].get(stage_id, None) is not None):
                current_out = self._prompt_decoder_sam(
                    frame_idx=stage_id,
                    is_init_cond_frame=is_init,
                    current_vision_feats=current_vision_feats, #current_vision_feats_inputs,
                    feat_sizes=feat_sizes,
                    point_inputs=backbone_out["point_inputs_per_frame"].get(t, None),
                    mask_inputs=backbone_out["mask_inputs_per_frame"].get(t, None),
                    gt_masks=backbone_out["gt_masks_per_frame"].get(t, None),
                    frames_to_add_correction_pt=frames_to_add_correction_pt,
                    frame_embedding=None,   #doesn't care about the frame embedding if mask input is provided
                    selector=selector,
                )
                
                raw_masks = current_out["pred_masks_high_res"]
                # raw_masks = current_out["pred_masks_high_res"]

                prob = torch.sigmoid(raw_masks.permute(1, 0, 2, 3))
                # masks = prob
                logits_cat, prob_cat = aggregate(prob, dim=1, return_logits=True)
                #strip away the background background
                masks = prob_cat[:, 1:]
                # masks = current_out["pred_masks_high_res_xmem"]
                #go from torch.Size([3, 1, 1024, 1024]) to torch.Size([1, 3, 1024, 1024]) with mask
                #masks_exp = masks.permute(1, 0, 2, 3)

                self.actual_obj_pointer = current_out["obj_ptr_xmem"]


            else:   #here for input_point or no input 
                # 2. select XMem references
                num_encoded = values.shape[3]
                ref_keys, ref_shrinkage, ref_values = self._select_xmem_refs(key, shrinkage, values, num_encoded, self.memory_cfg["num_ref_frames"], filler_one, B)
                #with C_slots)=1, we have the shape;
                #key.shape=torch.Size([1, 64, 8, 64, 64]), shrinkage.shape=torch.Size([1, 1, 8, 64, 64]), values.shape=torch.Size([1, 1, 512, 2, 64, 64])
                #ref_keys.shape=torch.Size([1, 64, 1, 64, 64]), ref_shrinkage.shape=torch.Size([1, 1, 1, 64, 64]), ref_values.shape=torch.Size([1, 1, 512, 2, 64, 64])
                
                #ey[:, :, t].shape=torch.Size([1, 64, 64, 64]), selection[:, :, t].shape=torch.Size([1, 64, 64, 64])
                
                # 3. Read from memory
                memory_readout = self.xmem.read_memory(key[:, :, t], selection[:, :, t] if selection is not None else None, 
                                ref_keys, ref_shrinkage, ref_values)
                
                # if self.actual_obj_pointer is not None:
                #     memory_readout = self.xmem.adapter_obj_pointer(self.actual_obj_pointer, memory_readout)
                
                # 4. update the hidden state with the gru 
                #memory_readout.shape=torch.Size([B, O, 512, 64, 64])
                hidden, frame_embedding = self.xmem.update_hidden_state(f16=f16[:,t], f8=f8[:,t], f4=f4[:,t], hidden_state=hidden, memory_readout=memory_readout, h_out=(t < (num_frames-1)))
                #hidden.shape=torch.Size([B, O, 64, 64, 64])
                #BEFORE: frame_embedding shape is torch.Size([B, O, 512, 64, 64]) where O is the number of objects and B is the batch size

                # 5. fuse the memory readout and the hidden state
                #pos16.shape = torch.Size([B, 256, 64, 64])
                # pos16_exp = pos16[:,t].squeeze(0).expand(num_objects, -1, -1, -1)
                # pos16_rep = pos16[:, t].repeat_interleave(C_slots, dim=0)
                frame_embedding = self.xmem.adapter_layer(frame_embedding.flatten(0, 1))
                # frame_embedding = self.xmem.adapter_layer(frame_embedding.squeeze(0))
                #AFTER: frame_embedding.shape=torch.Size([O, 256, 64, 64])

                #Note : shape of frame_embedding needs to be torch.Size([B*O, 256, 64, 64])
                # 6. call the prompt decoder function of sam2 to produce masks
                current_out = self._prompt_decoder_sam(
                    frame_idx=stage_id,
                    is_init_cond_frame=is_init,
                    current_vision_feats=current_vision_feats,
                    feat_sizes=feat_sizes,
                    point_inputs=backbone_out["point_inputs_per_frame"].get(t, None),
                    mask_inputs=backbone_out["mask_inputs_per_frame"].get(t, None),
                    gt_masks=backbone_out["gt_masks_per_frame"].get(t, None),
                    frames_to_add_correction_pt=frames_to_add_correction_pt,
                    frame_embedding=frame_embedding, # + pos16_exp, #frame_embedding 
                    selector=selector,
                )

                # 7. extract the masks from current_out
                # raw_masks = current_out["pred_masks_high_res"]
                raw_masks = current_out["pred_masks_high_res_xmem"]
                # import ipdb; ipdb.set_trace() 
                # prob = torch.sigmoid(raw_masks.permute(1, 0, 2, 3))
                raw_masks = raw_masks * selector 
                prob = torch.sigmoid(raw_masks)
                logits_cat, prob_cat = aggregate(prob, dim=1, return_logits=True)
                #strip away the background background
                masks = prob_cat[:, 1:]

                #store the last object pointer 
                self.actual_obj_pointer = current_out["obj_ptr_xmem"]

                # 8. encode the value 
                #should_encode = (not is_init) and (not is_last)
                if not is_last:
                    is_deep_update = torch.rand(1, device=key.device) < self.memory_cfg["deep_update_prob"]
                    v16, hidden = self.xmem.encode_value(rgbs[:, t], f16[:, t], hidden, masks, is_deep_update=is_deep_update)

                    # logits = current_out["object_score_logits_xmem"].squeeze(-1)
                    # is_visible = (logits > 0).float()
                    # is_occluded = (1.0 - is_visible).view(1, v16.shape[1], 1, 1, 1)

                    # v16 = self.xmem.occlusion_score_embedding(v16, is_occluded)

                    values = torch.cat([values, v16.unsqueeze(3)], 3)

            
            # 9. append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        # 10. output the final results
        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs


    def make_xmem_input_from_sam2_batch(self, batch: BatchedVideoDatapoint, C: int):
        """
        Given a BatchedVideoDatapoint (from the SAM2 data loader), produce the “data” dictionary
        that XMem’s training code expects.  In particular:
        - batch.img_batch:         [T, B, 3, H, W]
        - batch.masks:             [T, O, H, W]
        - batch.obj_to_frame_idx:  [T, O, 2]   (each row is (frame_idx, video_idx))
        and we output:
        - rgb: [B, T, 3, H, W]
        - first_frame_gt: [B, 1, C, H, W]
        - selector: [B, C, 1, 1]
        - num_filled_objects: length-B tensor with counts per video
        """
        # Unpack shapes
        T, B, Cin, H, W = batch.img_batch.shape   # [T, B, 3, H, W]
        O = batch.masks.shape[1]                 # total slots in the Sam2 batch

        # 1) Build rgb: [B, T, 3, H, W]
        rgb = batch.img_batch.permute(1, 0, 2, 3, 4).contiguous()

        # 2) Count how many objects each video has *at frame 0*
        #    frame0_pairs: [O, 2], each row = (frame_idx=0, video_idx)
        frame0_pairs = batch.obj_to_frame_idx[0]
        num_filled = [0] * B
        for obj_slot in range(O):
            _, b_idx = frame0_pairs[obj_slot].tolist()
            num_filled[b_idx] += 1
        num_filled_objects = torch.tensor(num_filled, dtype=torch.int64, device=rgb.device)

        # 3) Sanity check: each video must fit within C slots
        max_filled = int(num_filled_objects.max().item())
        if max_filled > C:
            raise ValueError(
                f"Video with index {int(num_filled_objects.argmax().item())} "
                f"has {max_filled} objects but XMem was configured for C={C} slots."
            )

        # 4) Build first_frame_gt: [B, 1, C, H, W], zero-padded
        first_frame_gt = torch.zeros((B, 1, C, H, W),
                                    dtype=torch.float32,
                                    device=rgb.device)
        # Copy each video’s true masks into its first_frame_gt channels
        # frame0_pairs: [O, 2], each row = (frame_idx=0, video_idx)
        for b in range(B):
            # find which global slots belong to video b
            slots_for_b = (frame0_pairs[:, 1] == b).nonzero(as_tuple=False).squeeze(-1)
            # now assign them into channels 0,1,... for this video
            for j, global_obj_slot in enumerate(slots_for_b.tolist()):
                first_frame_gt[b, 0, j] = batch.masks[0, global_obj_slot].float()

        # 5) Build selector: [B, C, 1, 1]
        selector = torch.zeros((B, C, 1, 1),
                            dtype=torch.float32,
                            device=rgb.device)
        for b in range(B):
            n = num_filled_objects[b].item()
            if n > 0:
                selector[b, :n, 0, 0] = 1.0

        return {
            "rgb": rgb,                           # [B, T, 3, H, W]
            "first_frame_gt": first_frame_gt,     # [B, 1, C, H, W]
            "selector": selector,                 # [B, C, 1, 1]
            "num_filled_objects": num_filled_objects  # length-B tensor
        }
           

    def reshape_feat(self, feat_raw, size, input):
        T, B, _, H_img, W_img = input.img_batch.shape
        h, w = size
        temp = feat_raw.permute(1, 2, 0).contiguous()
        feat = temp.view(B, T, temp.shape[1], h, w)
        return feat

    def _select_xmem_refs(
        self,
        key,             # [B, Ck, T, H, W]
        shrinkage,       # [B,  1, T, H, W] or None
        values,          # [B, Cv, T, 1, H, W]
        t,               # current time‐step (int)
        num_ref_frames,  # how many total ref frames to keep
        filler_one,      # tensor([0], dtype=torch.long, device=...)
        B                # batch size
    ):
        device = key.device
        # 1) If we’re still in the “warm-up” (t ≤ num_ref_frames),
        #    just take all history at once:
        if t <= num_ref_frames:
            ref_keys      = key[:, :, :t]                   # [B, Ck, t, H, W]
            ref_values    = values                           # [B, Cv, t, H, W]
            ref_shrinkage = shrinkage[:, :, :t] if shrinkage is not None else None
        else:
            # 2) Otherwise, sample exactly `num_ref_frames`:
            #    always include frame 0 (filler_one), plus (num_ref_frames-1)
            #    random picks from [1 .. t-1], per sample
            indices = [
                torch.cat([filler_one, torch.randperm(t-1)[:num_ref_frames-1]+1])
            for _ in range(B)]
            ref_values = torch.stack([
                values[bi, :, :, indices[bi]] for bi in range(B)
            ], 0)
            ref_keys = torch.stack([
                key[bi, :, indices[bi]] for bi in range(B)
            ], 0)
            ref_shrinkage = torch.stack([
                shrinkage[bi, :, indices[bi]] for bi in range(B)
            ], 0) if shrinkage is not None else None

        return ref_keys, ref_shrinkage, ref_values


    def _prompt_decoder_sam(self, 
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        mask_inputs,
        frame_embedding,
        selector=None,
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
        current_vision_pos_embeds=None,  # Optional
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        
        ########################################## all this part comes from _track_step function
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            pix_feat = frame_embedding #this is the the fusion of the memory readout and the hidden state
            
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,  
                point_inputs=point_inputs,   
                mask_inputs=mask_inputs,     
                high_res_features=high_res_features, 
                multimask_output=multimask_output, #True (that is a boolean)
            )
        ##########################################

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits

        # selector_mask = selector.squeeze(0).squeeze(-1).squeeze(-1).bool()  # Shape: [C_slots]
        # C_slots = self.memory_cfg["max_num_objects"]

        # ##from here I have a problem when using batch_size > 1

        # if selector_mask.sum().item()<C_slots:
        #     for k, v in list(current_out.items()):  # Use list() to allow safe modification
        #         if isinstance(v, list):
        #             if len(v) == 0:
        #                 continue  # nothing to do
        #             elif len(v) == C_slots and isinstance(v[0], torch.Tensor):
        #                 # List of per-object tensors
        #                 current_out[k] = [x.to(x.device) for x, keep in zip(v, selector_mask) if keep]
        #             elif isinstance(v[0], torch.Tensor) and v[0].shape[0] == C_slots:
        #                 # List of tensors shaped [C_slots, ...]
        #                 current_out[k] = [x[selector_mask].to(x.device) for x in v]
        #         elif isinstance(v, torch.Tensor):
        #             if v.shape[0] == C_slots:
        #                 current_out[k] = v[selector_mask].to(v.device)
        
            
        B, C_slots = selector.shape[:2]  # [B, C_slots, 1, 1]
        HW = high_res_masks.shape[2:]  # [1024, 1024] or whatever
        channels = high_res_masks.shape[1]  # typically 1
        
        # B, C_slots = selector.shape[:2]  # [B, C_slots, 1, 1]
        mask_flat = selector.reshape(-1).bool()   # [B, C_slots]

        for k, v in list(current_out.items()):
            if k == "pred_masks_high_res_xmem":
                # Skip the high-res masks, they are already reshaped above
                continue
            # (1) Singleton-list case
            elif isinstance(v, list) and len(v) == 1 and torch.is_tensor(v[0]) and v[0].shape[0] == B*C_slots:
                t = v[0][mask_flat]
                current_out[k] = [t] if t.ndim > 0 else [t.unsqueeze(0)]
            # (2) Tensor of shape [B*C_slots, ...]
            elif torch.is_tensor(v) and v.shape[0] == B*C_slots:
                t = v[mask_flat]
                current_out[k] = t
            # (3) Flat list of length B*C_slots
            elif isinstance(v, list) and len(v) == B*C_slots and all(torch.is_tensor(x) for x in v):
                filtered_list = [x for x, m in zip(v, mask_flat) if m]
                current_out[k] = filtered_list
            # (4) List-of-lists case (rare, but e.g. IOUs or multimasks)
            elif (isinstance(v, list) and len(v) == 1 and isinstance(v[0], list)
                and len(v[0]) == B*C_slots and all(torch.is_tensor(x) for x in v[0])):
                filtered_list = [x for x, m in zip(v[0], mask_flat) if m]
                current_out[k] = [filtered_list]

        high_res_masks_xmem = high_res_masks.view(B, C_slots, channels, *HW).squeeze(2)
        current_out["pred_masks_high_res_xmem"] = high_res_masks_xmem
        current_out["object_score_logits_xmem"] = object_score_logits
        current_out["obj_ptr_xmem"] = obj_ptr

        return current_out
