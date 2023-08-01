import torch.utils.data as data
from torch.utils.data import Sampler, WeightedRandomSampler
import torch 
import random 
import numpy as np 

class Dataset(data.Dataset):

    def __init__(self, data, target, static, stayid):
        """
            data: (n, 48, 182),
            target: (n, 48, 1),
            static: (n, 25),
            stayid: (n, 1)
        """

        # self.ti_data = ti_data
        self.data = data
        self.target = target
        self.static = static 
        self.stayid = stayid

    def __getitem__(self, index):

        data, static, target, stayid = self.data[index], self.static[index], self.target[index], self.stayid[index]

        data = np.float32(data)
        target = np.float32(target)
        static = np.float32(static)
        
        return data, static, target, stayid

    def __len__(self):
        return len(self.target)

def col_fn(batchdata):
    """
    A simple collate fn works like this: 
        def my_collate(batch):
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            target = torch.LongTensor(target)
            return [data, target]
        batchdata is a list of (data, static, target, stayid), which is picked by the custom sampler we wrote above
        [[(200, 48), (25), (48, 1), (1)], [(200, 28), (25), (48,1), (1)], [(200, 100), (25), (48, 1), (1)] ....]
    """

    len_data = len(batchdata)  
    seq_len = [batchdata[i][0].shape[-1] for i in range(len_data)]
    # [(48, ), (28, ), (100, )....]
    len_tem = [np.zeros((batchdata[i][0].shape[-1])) for i in range(len_data)]
    max_len = max(seq_len)

    # [(200, 48) ---> (200, 100)]
    padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len-batchdata[i][0].shape[-1])), \
                mode='constant', constant_values=-3) for i in range(len_data)]
#     # [(48, 1) ---> (100, 1)]
#     padded_label = [np.pad(batchdata[i][2], pad_width=((0, max_len-batchdata[i][0].shape[-1]), (0, 0)), \
#                 mode='constant', constant_values=0) for i in range(len_data)]
    label =  [batchdata[i][2] for i in range(len_data)]
    static = [batchdata[i][1] for i in range(len_data)]
    stayids = [batchdata[i][3] for i in range(len_data)]
    
    # [(48, ) ---> (100, )]
    mask = [np.pad(len_tem[i], pad_width=((0, max_len-batchdata[i][0].shape[-1])), \
            mode='constant', constant_values=1) for i in range(len_data)]
        
    return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.stack(static)), torch.from_numpy(np.stack(label)), torch.from_numpy(np.asarray(stayids)), torch.from_numpy(np.stack(mask))

def get_data_loader(args, train_head, dev_head, test_head, 
                train_sofa_tail, dev_sofa_tail, test_sofa_tail, 
                train_static = None, dev_static = None, test_static = None, 
                train_id = None, dev_id = None, test_id = None):
    """
    Args:
        args: main arguments
        train_head: list of train head data, e.g. [(200, 48), (200, 15), (200, 9), ...]
        dev_head: list of dev head data, 
        test_head: list of test head data,
        train_sofa_tail: list of tail part SOFA target. 
    """
    
    train_dataset = Dataset(train_head, train_sofa_tail, static = train_static, stayid = train_id)
    val_dataset = Dataset(dev_head, dev_sofa_tail, static = dev_static, stayid = dev_id)
    test_dataset = Dataset(test_head, test_sofa_tail, static = test_static, stayid = test_id)

    batch_sizes= args.bs
    val_batch_sizes = args.bs
    test_batch_sizes = args.bs
    # batch_sizes could be class 'numpy.int64'
    if not isinstance(batch_sizes, int):
        batch_sizes = batch_sizes.item()
        val_batch_sizes = val_batch_sizes.item()
        test_batch_sizes = test_batch_sizes.item()

    ctype, count= np.unique(train_sofa_tail, return_counts=True)
    total_samples = len(train_sofa_tail)
    weights_per_class = [total_samples / k / len(ctype) for k in count]
    weights = [weights_per_class[int(train_sofa_tail[i])] for i in range(int(total_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(total_samples))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_sizes, 
                            sampler = sampler, collate_fn=col_fn,
                            drop_last=False, pin_memory=False)

    dev_dataloader = data.DataLoader(val_dataset, batch_size=val_batch_sizes, 
                            collate_fn=col_fn,
                            drop_last=False, pin_memory=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=test_batch_sizes, 
                                collate_fn=col_fn,
                            drop_last=False, pin_memory=False)
        
    return train_dataloader, dev_dataloader, test_dataloader
