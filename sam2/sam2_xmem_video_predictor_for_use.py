# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

##
from omegaconf import OmegaConf 

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from sam2.modeling.sam2_base import NO_OBJ_SCORE
from sam2.modeling.sam2_base_no_mem import SAM2BaseXMem
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames

# from sam2.xmem_modules.modules import XMemFeatureProjector
# from sam2.xmem_modules.modules import GRU_Update
# from sam2.xmem_modules.modules import XMemValueEncoder

from sam2.xmem_modules.XMem_modules import XMem

from sam2.xmem_modules.memory_modules import XMemMemoryModule
from sam2.xmem_modules.memory_util import should_enable_lt

from sam2.xmem_modules.data.mask_wrapper import MaskMapper
from sam2.xmem_modules.data.preprocess import preprocess_mask_frame, reshape_feat

from sam2.xmem_modules.model.aggregate import aggregate


class SAM2XMemVideoPredictor(SAM2BaseXMem):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        xmem_style=False,
        **kwargs,
    ):

        #pull out the 'memory' section from the YAML (if any) and turn it into a *plain dict* for XMem.
        memory_cfg = kwargs.pop("memory", {})          # defaults to {}
        if not isinstance(memory_cfg, dict):           # Hydra DictConfig?
            memory_cfg = OmegaConf.to_container(memory_cfg, resolve=True)

        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

        self.memory_cfg = memory_cfg
        #self.memory_module = XMemMemoryModule(memory_cfg)
        
        self.xmem = XMem(self.memory_cfg, self.hidden_dim, self.sam_prompt_embed_dim)
        # self.xmem_value_encoder = self.xmem.xmem_value_encoder
        # self.xmem_projector = self.xmem.xmem_projector
        # self.sensory_gru = self.xmem.sensory_gru

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        #new dictionnary to store the last object pointer per object
        inference_state["last_obj_ptr_per_obj"] = {}

        ##add here enable_long_term_count_usage flag
        self.memory_cfg["enable_long_term_count_usage"] = should_enable_lt(inference_state["num_frames"], self.memory_cfg)

        ##fresh memory module for this video
        self.memory_module = XMemMemoryModule(self.memory_cfg)

        #initialiaze the mask wrapper
        self.mask_mapper = MaskMapper()

        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2VideoPredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        return sam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # We always allow adding new objects (including after tracking starts).
        allow_new_object = True
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["frames_tracked_per_obj"][obj_idx] = {}
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask_xmem_style_single_object(self, inference_state, frame_idx, input_mask, object_id):
        """
        Injects a ground-truth mask at frame_idx into XMem memory,
        without running any decoder or yielding—just the GT update path.
        """
        #Preprocess Part 
        import ipdb; ipdb.set_trace()
        internal_id = object_id 
        self.mask_mapper.remappings[object_id] = internal_id

        mask = input_mask
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2

        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])
            # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig
        
        masks = mask_inputs  # [1, 1, H, W]
        internal_labels = (self.mask_mapper.remappings[object_id])
        
        valid_labels = range(1, 2)
        self.memory_module.set_all_labels(valid_labels)

        # # mask_info = preprocess_mask_frame(frame_idx, frame_names, input_mask_dir, video_name, self.mask_mapper, self.memory_module, inference_state["device"], self.image_size)
        # mask_info = preprocess_mask_frame(frame_idx, frame_names, input_mask_dir, video_name, self.mask_mapper, self.memory_module, inference_state["device"], self.image_size, object_ids)
        # mask         = mask_info["mask"]    # [N, H, W] float tensor on GPU --> shape : torch.Size([2, 1280, 720])
        # valid_labels = mask_info["labels"]  # e.g. [1,2,3]
        
        ##populate inference_state dictionary 
        internal_to_original = {v: k for k, v in self.mask_mapper.remappings.items()}

                # Initialize per-object data
        for obj_idx, internal_id in enumerate(valid_labels):
            original_id = internal_to_original[internal_id]

            # Populate the ID mappings
            inference_state["obj_id_to_idx"][original_id] = internal_id
            inference_state["obj_idx_to_id"][internal_id] = original_id
            inference_state["obj_ids"].append(original_id)

            # Add the mask for this object
            mask_i = masks[obj_idx]  # [1, 1, H, W] float tensor on GPU
            inference_state["mask_inputs_per_obj"][obj_idx] = {frame_idx: mask_i}

        # 1. Advance XMem’s timestep
        self.memory_module.curr_ti += 1
        gt_mask_given = True

        is_last = (self.memory_module.curr_ti == inference_state["num_frames"] - 1)

        is_mem, is_deep, is_norm, need_seg = self.memory_module.compute_flags(
            self.memory_module.curr_ti, self.memory_module.last_mem_ti, self.memory_module.last_deep_update_ti,
            self.memory_module.mem_every, self.memory_module.deep_update_every, self.memory_module.deep_update_sync,
            gt_mask_given, is_last, self.memory_module.all_labels, valid_labels)

        # 2. Load & pad the image for this frame
        img = inference_state["images"][frame_idx]         # already a torch.Tensor --> shape torch.Size([3, 1024, 1024])
        # img, pad = pad_divide_by(img, 16)
        img = img.unsqueeze(0)                             # batch dim (for later use in value encoder)

        # 4. extract features once for *all* objects on frame f
        (_, _, current_vision_feats, current_pos, feat_sizes) = self._get_image_feature(inference_state, frame_idx, 1) #mask.shape[0]
        f16 = reshape_feat(current_vision_feats[-1], feat_sizes[-1]) 

        # 5. project features → (K,S,E) for XMem memory (use the fused stride‑16 feature map) and memory read per object
        #key, shrinkage, selection = self.xmem_projector(f16, need_sk=True, need_ek=self.memory_cfg['enable_long_term']) #k,s,e
        key, shrinkage, selection = self.xmem.key_projection(img, f16, need_sk=True, need_ek=self.memory_cfg['enable_long_term']) #k,s,e

        # 6. Initialize sensory memory (hidden) for each object
        #    `valid_labels` is your continuous label list from MaskMapper, e.g. [1,2,3]
        self.memory_module.memory.create_hidden_state(len(self.memory_module.all_labels), key)

        # 7. pass mask inputs through the SAM prompt decoder
        pred_masks_per_obj = [None] * len(valid_labels)  
        
        current_out = self._prompt_decoder_sam(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=masks, # [1, 1, H, W] float tensor on GPU
            frame_embedding=None,
        )
        # Extract predicted high-res mask for this object:
        high_res_masks = current_out["pred_masks_high_res"]  # shape [O_batch, 1, H, W] or [1,1,H,W] when batched per-object
        # object_id = inference_state["obj_id_to_idx"][obj_idx] #not really needed
        # Since we ran per-object, shape should be [1,1,H, W]
        # Move to GPU if needed:
        pred_masks_per_obj[obj_idx] = high_res_masks  # keep for yielding

        obj_ptr = current_out["obj_ptr"] 
        inference_state["last_obj_ptr_per_obj"][obj_idx] = obj_ptr 

        # 7. Concatenate all predicted masks across objects
        if len(pred_masks_per_obj)>1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]

        # 7. Build value prompt via soft‐aggregation
        # gt_prob_mask = aggregate(mask, dim=0)           # [N+1, H, W]
        # value_prompt = gt_prob_mask[1:].unsqueeze(0)    # [1, N, H, W]

        prob = torch.sigmoid(all_pred_masks.permute(1, 0, 2, 3))
        # pred_masks = prob
        logits_cat, prob_cat = aggregate(prob, dim=1, return_logits=True)
        pred_masks = prob_cat[:, 1:]

        # 8. Encode (value, new_hidden) using XMem’s value‐encoder
        #value, new_hidden = self.value_encoder.encode_value(img, f16, self.memory_module.memory.get_hidden(), value_prompt, is_deep_update=is_deep)
        value, new_hidden = self.xmem.encode_value(img, f16, self.memory_module.memory.get_hidden(), pred_masks, is_deep_update=is_deep)
        #note : first‐frame should always deep‐update

        # logits = current_out["object_score_logits"].squeeze(-1)
        # is_visible = (logits > 0).float()
        # is_occluded = (1.0 - is_visible).view(1, value.shape[1], 1, 1, 1)

        # value = self.xmem.occlusion_score_embedding(value, is_occluded)

        # 9. Write into both working/long‐term memory
        self.memory_module.memory.add_memory(
                        key,        # [1, CK, h, w]
                        shrinkage,   # [1, 1, h, w]               
                        value,       # [1, N, CV, h, w]
                        self.memory_module.all_labels, 
                        selection=selection if self.memory_cfg['enable_long_term'] else None) 

        # 10. Update the XMem timestamps
        self.memory_module.last_mem_ti = self.memory_module.curr_ti

        # 11. save the updated hidden
        if is_deep:
            self.memory_module.memory.set_hidden(new_hidden)
            self.memory_module.last_deep_update_ti = self.memory_module.curr_ti

        # 12. reshape to fit original image size s
        #don't forget to permute dim=0 and dim=1 of the value prompt : go from [1, N, H, W] to [N, 1, H, W] --> value_prompt.permute(1, 0, 2, 3)
        _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
        # import ipdb; ipdb.set_trace()
        return frame_idx, inference_state["obj_ids"], video_res_masks

    @torch.inference_mode()
    def add_new_mask_xmem_style(self, inference_state, frame_idx, object_ids, per_obj_input_mask):
        """
        Injects a ground-truth mask at frame_idx into XMem memory,
        without running any decoder or yielding—just the GT update path.
        """
        # import ipdb; ipdb.set_trace()
        #Preprocess Part 
        internal_labels = []
        masks = []  
        for internal_id, (obj_id, mask) in enumerate(per_obj_input_mask.items(), start=1): #note: actually mask is NPY format !!!
            self.mask_mapper.remappings[obj_id] = internal_id

            # mask = object_mask_list[obj_id]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            assert mask.dim() == 2

            mask_H, mask_W = mask.shape
            mask_inputs_orig = mask[None, None]  # add batch and channel dimension
            mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])
                # resize the mask if it doesn't match the model's image size
            if mask_H != self.image_size or mask_W != self.image_size:
                mask_inputs = torch.nn.functional.interpolate(
                    mask_inputs_orig,
                    size=(self.image_size, self.image_size),
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
                mask_inputs = (mask_inputs >= 0.5).float()
            else:
                mask_inputs = mask_inputs_orig
            
            masks.append(mask_inputs)  # [1, 1, H, W]
            internal_labels.append(self.mask_mapper.remappings[obj_id])
        
        valid_labels = range(1, len(internal_labels) + 1)
        self.memory_module.set_all_labels(valid_labels)

        # # mask_info = preprocess_mask_frame(frame_idx, frame_names, input_mask_dir, video_name, self.mask_mapper, self.memory_module, inference_state["device"], self.image_size)
        # mask_info = preprocess_mask_frame(frame_idx, frame_names, input_mask_dir, video_name, self.mask_mapper, self.memory_module, inference_state["device"], self.image_size, object_ids)
        # mask         = mask_info["mask"]    # [N, H, W] float tensor on GPU --> shape : torch.Size([2, 1280, 720])
        # valid_labels = mask_info["labels"]  # e.g. [1,2,3]
        
        ##populate inference_state dictionary 
        internal_to_original = {v: k for k, v in self.mask_mapper.remappings.items()}
                # Initialize per-object data
        for obj_idx, internal_id in enumerate(valid_labels):
            original_id = internal_to_original[internal_id]

            # Populate the ID mappings
            inference_state["obj_id_to_idx"][original_id] = internal_id
            inference_state["obj_idx_to_id"][internal_id] = original_id
            inference_state["obj_ids"].append(original_id)

            # Add the mask for this object
            mask_i = masks[obj_idx]  # [1, 1, H, W] float tensor on GPU
            inference_state["mask_inputs_per_obj"][obj_idx] = {frame_idx: mask_i}

        # 1. Advance XMem’s timestep
        self.memory_module.curr_ti += 1
        gt_mask_given = True

        is_last = (self.memory_module.curr_ti == inference_state["num_frames"] - 1)

        is_mem, is_deep, is_norm, need_seg = self.memory_module.compute_flags(
            self.memory_module.curr_ti, self.memory_module.last_mem_ti, self.memory_module.last_deep_update_ti,
            self.memory_module.mem_every, self.memory_module.deep_update_every, self.memory_module.deep_update_sync,
            gt_mask_given, is_last, self.memory_module.all_labels, valid_labels)

        # 2. Load & pad the image for this frame
        img = inference_state["images"][self.memory_module.curr_ti]         # already a torch.Tensor --> shape torch.Size([3, 1024, 1024])
        # img, pad = pad_divide_by(img, 16)
        img = img.unsqueeze(0)                             # batch dim (for later use in value encoder)

        # 4. extract features once for *all* objects on frame f
        (_, _, current_vision_feats, current_pos, feat_sizes) = self._get_image_feature(inference_state, frame_idx, 1) #mask.shape[0]
        f16 = reshape_feat(current_vision_feats[-1], feat_sizes[-1]) 

        # 5. project features → (K,S,E) for XMem memory (use the fused stride‑16 feature map) and memory read per object
        #key, shrinkage, selection = self.xmem_projector(f16, need_sk=True, need_ek=self.memory_cfg['enable_long_term']) #k,s,e
        key, shrinkage, selection = self.xmem.key_projection(img, f16, need_sk=True, need_ek=self.memory_cfg['enable_long_term']) #k,s,e

        # 6. Initialize sensory memory (hidden) for each object
        #    `valid_labels` is your continuous label list from MaskMapper, e.g. [1,2,3]
        self.memory_module.memory.create_hidden_state(len(self.memory_module.all_labels), key)

        # 7. pass mask inputs through the SAM prompt decoder
        pred_masks_per_obj = [None] * len(object_ids)
        # pred_occ_scores_per_obj = [None] * len(object_ids)  # for occlusion scores  
        # import ipdb; ipdb.set_trace()
        for obj_idx in inference_state["obj_ids"]:

            current_out = self._prompt_decoder_sam(
                frame_idx=frame_idx,
                is_init_cond_frame=False,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=masks[obj_idx], # [1, 1, H, W] float tensor on GPU
                frame_embedding=None,
            )
            # Extract predicted high-res mask for this object:
            high_res_masks = current_out["pred_masks_high_res"]  # shape [O_batch, 1, H, W] or [1,1,H,W] when batched per-object
            # object_id = inference_state["obj_id_to_idx"][obj_idx] #not really needed
            occlusion_score = current_out["object_score_logits"]
            # Since we ran per-object, shape should be [1,1,H, W]
            # Move to GPU if needed:
            pred_masks_per_obj[obj_idx] = high_res_masks  # keep for yielding
            # pred_occ_scores_per_obj[obj_idx] = occlusion_score  # keep for yielding

            obj_ptr = current_out["obj_ptr"] 
            inference_state["last_obj_ptr_per_obj"][obj_idx] = obj_ptr 

        # 7. Concatenate all predicted masks across objects
        if len(pred_masks_per_obj)>1:
            all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            # all_pred_occ_scores = torch.cat(pred_occ_scores_per_obj, dim=0)
        else:
            all_pred_masks = pred_masks_per_obj[0]
            # all_pred_occ_scores = pred_occ_scores_per_obj[0]

        # 7. Build value prompt via soft‐aggregation
        prob = torch.sigmoid(all_pred_masks.permute(1, 0, 2, 3))
        # pred_masks = prob
        logits_cat, prob_cat = aggregate(prob, dim=1, return_logits=True)
        pred_masks = prob_cat[:, 1:]

        # 8. Encode (value, new_hidden) using XMem’s value‐encoder
        value, new_hidden = self.xmem.encode_value(img, f16, self.memory_module.memory.get_hidden(), pred_masks, is_deep_update=is_deep)
        #note : first‐frame should always deep‐update
        
        # logits = all_pred_occ_scores.squeeze(-1)
        # is_visible = (logits > 0).float()
        # is_occluded = (1.0 - is_visible).view(1, value.shape[1], 1, 1, 1)
        # value = self.xmem.occlusion_score_embedding(value, is_occluded)

        # 9. Write into both working/long‐term memory
        self.memory_module.memory.add_memory(
                        key,        # [1, CK, h, w]
                        shrinkage,   # [1, 1, h, w]               
                        value,       # [1, N, CV, h, w]
                        self.memory_module.all_labels, 
                        selection=selection if self.memory_cfg['enable_long_term'] else None) 

        # 10. Update the XMem timestamps
        self.memory_module.last_mem_ti = self.memory_module.curr_ti

        # 11. save the updated hidden
        if is_deep:
            self.memory_module.memory.set_hidden(new_hidden)
            self.memory_module.last_deep_update_ti = self.memory_module.curr_ti

        # 12. reshape to fit original image size s
        #don't forget to permute dim=0 and dim=1 of the value prompt : go from [1, N, H, W] to [N, 1, H, W] --> value_prompt.permute(1, 0, 2, 3)
        _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
        return frame_idx, inference_state["obj_ids"], video_res_masks


    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask

        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Check and make sure that every object has received input points or masks.
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError(
                "No input points or masks are provided for any object; please add inputs first."
            )

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                # Separately consolidate conditioning and non-conditioning temp outputs
                storage_key = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                # Find all the frames that contain temporary outputs for any objects
                # (these should be the frames that have just received clicks for mask inputs
                # via `add_new_points_or_box` or `add_new_mask`)
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder on the temporary outputs (if the memory feature is missing)
                    if out["maskmem_features"] is None:
                        high_res_masks = torch.nn.functional.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            high_res_masks=high_res_masks,
                            object_score_logits=out["object_score_logits"],
                            # these frames are what the user interacted with
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc

                    obj_output_dict[storage_key][frame_idx] = out
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )

                # clear temporary outputs in `temp_output_dict_per_obj`
                obj_temp_output_dict[storage_key].clear()

            # check and make sure that every object has received input points or masks
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(
                    f"No input points or masks are provided for object id {obj_id}; please add inputs first."
                )
            # edge case: if an output is added to "cond_frame_outputs", we remove any prior
            # output on the same frame in "non_cond_frame_outputs"
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    def propagate_in_video_preflight_xmem(self, inference_state):
        """
        Prepares inference state before propagation in an XMem-style system.
        Consolidates temporary outputs into permanent storage, clears memory if needed,
        and checks that each object has at least one conditioning frame.

        Assumes XMem-style memory updates are handled elsewhere (e.g., add_new_mask or step).
        """
        batch = self._get_obj_num(inference_state)
        if batch == 0:
            raise RuntimeError("No objects to propagate.")

        for obj_idx in range(batch):
            obj_out = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp = inference_state["temp_output_dict_per_obj"][obj_idx]

            # Consolidate temp outputs into main output dicts
            for is_cond in (False, True):
                key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                for t, out in obj_temp[key].items():
                    obj_out[key][t] = out
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(inference_state, t, obj_idx)
                obj_temp[key].clear()

            # Ensure at least one conditioning frame exists
            if len(obj_out["cond_frame_outputs"]) == 0:
                raise RuntimeError(f"Object {self._obj_idx_to_id(inference_state, obj_idx)} has no conditioning input.")

            # Remove duplicate frames from non_cond if they're already in cond
            for t in obj_out["cond_frame_outputs"]:
                obj_out["non_cond_frame_outputs"].pop(t, None)

    @torch.inference_mode()
    def propagate_in_video_xmem_style(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """
        Per-frame, per-object inference:
        For each frame f, we first extract shared image features.
        Then for each object slot obj_idx, we:
        - slice its hidden & memory readout,
        - update hidden,
        - fuse & decode with SAM separately,
        - encode predicted mask back into memory slot.
        """
        # 1. initialize some basic variables
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)  # number of object slots active
        # 2. decide frame range
        if start_frame_idx is None:
            start_frame_idx = min(
                frame_idx
                for obj_inputs in inference_state["mask_inputs_per_obj"].values()
                for frame_idx in obj_inputs.keys()
            ) + 1
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        # Loop over frames
        total_time_update_hidden = []
        # total_time_reshape_features = []
        # total_time_key_projection = []
        for f in tqdm(processing_order, desc="sam2-xmem-propagate"):
            # advance time index
            self.memory_module.curr_ti += 1
            is_last = (f == processing_order[-1])

            # load image tensor for this frame
            img = inference_state["images"][f]  # [3, H, W]
            img = img.unsqueeze(0)  # [1, 3, H, W]

            # determine XMem flags
            gt_mask_given = False
            valid_labels = None

            is_mem, is_deep, is_norm, need_seg = self.memory_module.compute_flags(
                self.memory_module.curr_ti, self.memory_module.last_mem_ti, self.memory_module.last_deep_update_ti,
                self.memory_module.mem_every, self.memory_module.deep_update_every, self.memory_module.deep_update_sync,
                gt_mask_given, is_last, self.memory_module.all_labels, valid_labels)

            # 4. extract shared image features once for *all* objects
            (_, _, current_vision_feats, current_pos, feat_sizes) = self._get_image_feature(inference_state, f, 1)


            (f4_raw, f8_raw, f16_raw) = current_vision_feats[-3:]
            f4 = reshape_feat(f4_raw, feat_sizes[-3])   # [1, C4, h4, w4]
            f8 = reshape_feat(f8_raw, feat_sizes[-2])   # [1, C8, h8, w8]
            f16 = reshape_feat(f16_raw, feat_sizes[-1]) # [1, C16, h16, w16]

            (pos4_raw, pos8_raw, pos16_raw) = current_pos[-3:]
            pos16 = reshape_feat(pos16_raw, feat_sizes[-1])  

            # 5. project features for XMem key/shrinkage once
            key, shrinkage, selection = self.xmem.key_projection(img, f16, need_sk=True, need_ek=self.memory_cfg['enable_long_term'])
            # key: [1, Ck, h16, w16]; shrinkage: [1,1,h16,w16] or None; selection used in memory.match_memory


            # If segmentation is needed this frame:
            if need_seg:
                # 6. get memory readout for all objects: returns [N, C, h16, w16]
                # match_memory(key, selection) → [N, C, h, w]
                memory_readout_all = self.memory_module.memory.match_memory(key, selection).unsqueeze(0) #shape: [1, batch_size, value_dim, h, w]

                # 7. get hidden state for all objects: [1, N, hidden_dim, h, w]
                hidden_all = self.memory_module.memory.get_hidden()  # shape [1, N, ...]


                # Prepare a container to collect per-object predicted masks for this frame
                # (you can yield immediately per object if preferred)
                # Here we collect to yield a dict of masks per object.
                pred_masks_per_obj = [None] * batch_size
                # pred_occ_scores_per_obj = [None] * batch_size  # for occlusion scores  
                # 8. Loop per object slot
                for obj_idx in range(batch_size):
                    # 8.1 slice hidden & memory_readout for this object
                    # memory_readout_all: [N, C, h, w], pick slot obj_idx
                    mr_obj = memory_readout_all[:, obj_idx].unsqueeze(0)
                    # → [1, 1, value_dim, h, w]
                    hidden_prev_obj = hidden_all[:, obj_idx:obj_idx+1]  # [1,1,hidden_dim,h,w]
                    # note: hidden_all is [1,N,...]; slicing keeps batch dim 1 and slot dim 1


                    # 8.2 update hidden via XMem GRU for this object, if needed
                    new_hidden_obj, frame_emb_obj = self.xmem.update_hidden_state(f16=f16, f8=f8, f4=f4, hidden_state=hidden_prev_obj, memory_readout=mr_obj, h_out=is_norm)  # → [1, 1, hidden_dim, h, w]


                    # 8.3 write back hidden if normalization flag
                    if is_norm:
                        # set_hidden expects full hidden_all; we can update only this slot:
                        # retrieve full hidden_all, replace slot, then set_hidden
                        hidden_all[:, obj_idx:obj_idx+1] = new_hidden_obj
                        self.memory_module.memory.set_hidden(hidden_all)



                    # 8.4 fuse memory_readout + hidden to get frame embedding for this object
                    # frame_emb_obj = self.xmem.fusion_memory_hidden_state(mr_obj, new_hidden_obj)  # [1, feat_dim, h, w]
                    frame_emb_obj = self.xmem.adapter_layer(frame_emb_obj.squeeze(0))
                    #use pos16 and add it to frame_emb_obj
                    # frame_emb_obj = frame_emb_obj + pos16  # [1, feat_dim, h, w] + [1, C_pos, h, w] 

                    # 8.5 call SAM decoder for this single object
                    # But since we run per-object, current_vision_feats are shared; SAM decoder expects dims [B=1, C, h, w]
                    current_out = self._prompt_decoder_sam(
                        frame_idx=f,
                        is_init_cond_frame=False,
                        current_vision_feats=current_vision_feats,
                        feat_sizes=feat_sizes,
                        point_inputs=None,
                        mask_inputs=None,
                        frame_embedding=frame_emb_obj, # + pos16, #frame_emb_obj
                    )


                    # Extract predicted high-res mask for this object:
                    high_res_masks = current_out["pred_masks_high_res"]  # shape [O_batch, 1, H, W] or [1,1,H,W] when batched per-object
                    #object_id = inference_state["obj_id_to_idx"][obj_idx] #not really needed
                    # Since we ran per-object, shape should be [1,1,H, W]
                    occlusion_score = current_out["object_score_logits"]
                    # Move to GPU if needed:
                    pred_masks_per_obj[obj_idx] = high_res_masks  # keep for yielding
                    # pred_occ_scores_per_obj[obj_idx] = occlusion_score  # keep for yielding


                if len(pred_masks_per_obj)>1:
                    all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                    # all_pred_occ_scores = torch.cat(pred_occ_scores_per_obj, dim=0)
                else:
                    all_pred_masks = pred_masks_per_obj[0]
                    # all_pred_occ_scores = pred_occ_scores_per_obj[0]


            prob = torch.sigmoid(all_pred_masks.permute(1, 0, 2, 3))
            # masks = prob
            logits_cat, prob_cat = aggregate(prob, dim=1, return_logits=True)
            masks = prob_cat[:, 1:]

            #save tensor of predicted masks in pt format
            # torch.save(masks.cpu(), f"./output_tensor_test_masks/masks_frame_{f*4}.pt")
         
            # 8.6 if memory write needed (is_mem), encode this object's mask back into XMem
            if is_mem:
                #img.shape = torch.Size([1, 3, 1024, 1024])
                #f16.shape = torch.Size([1, 256, 64, 64])
                #self.memory_module.memory.get_hidden().shape = torch.Size([1, 2, 64, 64, 64]) 
                #masks.shape = torch.Size([1, 2, 1024, 1024])
                # encode value & update hidden for deep update if needed
                value, hidden_from_val = self.xmem.encode_value(img, f16, self.memory_module.memory.get_hidden(), masks, is_deep_update=is_deep)
                # note: get_hidden() now yields updated hidden after GRU above
                # value_obj: [1,1,Cv,h,w]; hidden_from_val: [1,1,hidden,h,w]


                # logits = all_pred_occ_scores.squeeze(-1)
                # is_visible = (logits > 0).float()
                # is_occluded = (1.0 - is_visible).view(1, value.shape[1], 1, 1, 1)
                # value = self.xmem.occlusion_score_embedding(value, is_occluded)

                # write into memory for this object slot
                self.memory_module.memory.add_memory(
                    key,         # [1, Ck, h, w]
                    shrinkage,   # [1, 1, h, w]
                    value,       # [1, N, CV, h, w]
                    self.memory_module.all_labels, 
                    selection=selection if self.memory_cfg['enable_long_term'] else None)     

                self.memory_module.last_mem_ti = self.memory_module.curr_ti
                # update hidden if deep:
                if is_deep:
                    # fetch full hidden_all, replace this slot, then set
                    self.memory_module.memory.set_hidden(hidden_from_val)
                    self.memory_module.last_deep_update_ti = self.memory_module.curr_ti


            #15. reshape to fit original image size s
            _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
            frame_idx = f

            yield frame_idx, obj_ids, video_res_masks
        
        # print("Average time value encoder (is_mem):", np.mean(total_time_update_hidden))
        # print len of total_time_update_hidden
        # print("Number of frames processed:", len(total_time_update_hidden))
        # print("Average time key projection:", np.mean(total_time_key_projection))


    @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            # The frame is not a conditioning frame anymore since it's not receiving inputs,
            # so we "downgrade" its output (if exists) to a non-conditioning frame output.
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()

        ##reset the memory module
        self.memory_module.all_labels = None 
        self.memory_module.clear_memory()



    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["frames_tracked_per_obj"].values():
            v.clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features


    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])
        _map_keys(inference_state["frames_tracked_per_obj"])

        # Step 3: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_frame_outputs.pop(t, None)

    @torch.inference_mode()
    def _prompt_decoder_sam(self, 
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        mask_inputs,
        frame_embedding,
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

        return current_out

