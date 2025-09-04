import os
import torch
from PIL import Image
import numpy as np
from sam2.xmem_modules.data.mask_wrapper import MaskMapper

def load_and_resize_mask(mask_path: str, image_size: int) -> torch.Tensor:
    """
    Load a single‐frame indexed mask from disk, resize it to (image_size x image_size)
    using nearest‐neighbor (to preserve discrete labels), and return a H×W torch.Tensor.
    """
    # 1) load as grayscale PIL image
    mask_pil = Image.open(mask_path).convert("L")
    # 2) resize with nearest neighbor
    mask_resized_pil = mask_pil.resize((image_size, image_size), resample=Image.NEAREST)
    # 3) to numpy
    mask_np = np.array(mask_resized_pil, dtype=np.uint8)

    return mask_np

def reshape_feat(feat_raw, size):
    # feat_raw: [HW, 1, C], size=(H, W)
    hw, b1, c = feat_raw.shape
    h, w = size
    # permute from [HW, B, C] to [B, C, HW] then view→[B, C, H, W]
    return feat_raw.permute(1, 2, 0).view(1, c, h, w)


# def preprocess_mask_frame(frame_idx, frame_names, input_mask_dir, video_name, mapper, processor, device, image_size):
#     """
#     Preprocess the mask for the current frame by:
#     - Loading it from disk
#     - Converting to internal label format using MaskMapper
#     - Registering label mapping in the processor

#     Returns a dict with:
#         - "skip": whether to skip this frame
#         - "mask": tensor of shape [num_objects, H, W] or None
#         - "labels": list of internal object labels or None
#     """
#     #add "_merged" at the of input_mask_dir : 
#     input_mask_dir = input_mask_dir + "_merged"
#     frame_name = frame_names[frame_idx]
#     mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")

#     if not os.path.exists(mask_path):
#         print(f"⚠️ Skipping frame {frame_idx} — no mask found at {mask_path}")
#         return {"skip": True, "mask": None, "labels": None}
#     # Load the mask image
#     mask_img = Image.open(mask_path).convert("L")
#     #mask_np = np.array(mask_img, dtype=np.uint8)
#     mask_np = load_and_resize_mask(mask_path, image_size) 

#     #extract width and height from mask_np
#     mask_height, mask_width = mask_np.shape
    
#     # Convert to one-hot tensor and remap labels
#     # mask_tensor, labels = mapper.convert_mask(mask_np)
#     # mask_tensor = mask_tensor.to(device)
#     mask_tensor, labels = mapper.convert_mask(mask_np)
#     mask_tensor = mask_tensor.to(device)  # [N, H, W]
#     if (mask_height != image_size) or (mask_width != image_size):
#         mask_tensor = F.interpolate(
#                 mask_tensor, 
#                 size=(image_size, image_size),
#                 antialias=True,
#                 mode="bilinear",
#                 align_corners=False,
#             )
#     else:
#         mask_tensor = mask_tensor

#     # Register the remapped internal labels with the memory processor
#     processor.set_all_labels(list(mapper.remappings.values()))

#     return {"mask": mask_tensor, "labels": labels}


#################################################################################################3
def preprocess_mask_frame(
    frame_idx,
    frame_names,
    input_mask_dir,
    video_name,
    mapper,
    processor,
    device,
    image_size,
    object_id_set,  # e.g. {76, 29, 150}
):
    """
    Load per-object binary masks from subfolders (named by original IDs like '076').
    Builds [N, H, W] mask tensor. Assigns internal IDs starting from 1.
    Final remapping: {original_id: internal_id}, e.g., {76: 1, 29: 2}
    """

    frame_name = frame_names[frame_idx]
    video_obj_dir = os.path.join(input_mask_dir, video_name)

    masks = []
    internal_labels = []

    # Assign internal IDs to each original object ID
    for internal_id, obj_id in enumerate(sorted(object_id_set), start=1):
        mapper.remappings[obj_id] = internal_id

    for obj_id in sorted(object_id_set):
        folder_name = f"{int(obj_id):03d}"  # zero-padded folder name
        mask_path = os.path.join(video_obj_dir, folder_name, f"{frame_name}.png")

        mask_np = load_and_resize_mask(mask_path, image_size)

        mask_t = torch.from_numpy((mask_np > 0).astype(np.float32)).to(device)
        if mask_t.shape[-2:] != (image_size, image_size):
            mask_t = F.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)
        masks.append(mask_t.unsqueeze(0))  # [1, H, W]
        internal_labels.append(mapper.remappings[obj_id])


    mask_tensor = torch.cat(masks, dim=0)  # [N, H, W]
    labels = range(1, len(internal_labels) + 1)
    processor.set_all_labels(labels)

    return {"mask": mask_tensor, "labels": labels}
