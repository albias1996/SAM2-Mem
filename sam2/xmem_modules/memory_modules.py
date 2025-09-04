import torch
from sam2.xmem_modules.memory_manager import MemoryManager

class XMemMemoryModule:
    def __init__(self, config):
        self.config = config
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']


        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        else:
            # Define anyway to avoid AttributeError later, even if unused
            self.last_deep_update_ti = 0 
        self.memory = MemoryManager(config=self.config)

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    @staticmethod
    def compute_flags(curr_ti: int,
                      last_mem_ti: int,
                      last_deep_update_ti: int,
                      mem_every: int,
                      deep_update_every: int,
                      deep_update_sync: bool,
                      gt_mask_provided: bool,
                      end: bool,
                      all_labels=None,
                      valid_labels=None):
        """
        Returns (is_mem_frame, is_deep_update, is_normal_update, need_segment)
        – Logic identical to original XMem.
        """
        is_mem_frame = ((curr_ti - last_mem_ti) >= mem_every or gt_mask_provided) and (not end)

        if deep_update_sync:
            is_deep_update = is_mem_frame and (not end)
        else:
            is_deep_update = ((curr_ti - last_deep_update_ti) >= deep_update_every) and (not end)

        is_normal_update = (not deep_update_sync or not is_deep_update) and (not end)

        # if curr_ti == 0:
        #     need_segment = False  # no segmentation at t=0
        # elif valid_labels is None:
        #     need_segment = True
        # else:
        #     need_segment = (all_labels is None) or (len(all_labels) != len(valid_labels))
        need_segment = (curr_ti > 0) and ((valid_labels is None) or (len(all_labels) != len(valid_labels)))
            
        return is_mem_frame, is_deep_update, is_normal_update, need_segment
