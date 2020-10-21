import torch
from torch.utils.data import Dataset
import numpy as np


from skimage.transform import rotate
from scipy import ndimage
from skimage.util import img_as_ubyte

#jd's function
#class _image_aug:

def _image_aug(pic, angle, centroid_x=23, centroid_y=23, win=16, scale=1.45):
    im_sz = int(np.floor(pic.shape[1]*scale))
    pic_ = np.uint8(np.zeros((im_sz,im_sz,3),dtype=int))

    pic_[:,:,0] = ndimage.zoom(pic[0,:,:],scale)

    pic_[:,:,1] = ndimage.zoom(pic[1,:,:],scale)
    pic_[:,:,2] = ndimage.zoom(pic[2,:,:],scale)

    image_aug = rotate(pic_, angle, resize=False)
    #print(image_aug.shape)
    image_aug_ = image_aug[centroid_x-win:centroid_x+win,centroid_y-win:centroid_y+win,:]
    image_aug_ = image_aug_.reshape(3,32,32)

    return img_as_ubyte(image_aug_)

#jd's version to manipulate the data
class GetSlotDataset(Dataset):

    def __init__(self, datatset_to_process, slot, shift, type='train'):
        super().__init__()
        self.dataset = datatset_to_process
        self.indeces = []

        label = np.asarray([lbl for _,lbl in self.dataset])
        idx = np.asarray([np.where(label==i) for i in np.unique(label)])
        
        if type == 'train':
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][(slot-1)*50:slot*50]
                    )
                )
        else:
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][500:600]
                    )
                )
                
    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.dataset[self.indeces[index]]

#jd's version to randomly shuffle class labels for tasks 2-10
class GetShuffledDataset(Dataset):
    
    def __init__(self, datatset_to_process, slot, shift, type='train'):
        super().__init__()
        self.dataset = datatset_to_process
        self.indeces = []

        label = np.asarray([lbl for _,lbl in self.datatset])
        idx = [np.where(label==i) for i in np.unique(label)]

        shuffled_label = []
        for task in range(10):
            _tmp = [] 
            for cls in range(10*task,10*(task+1)):
                _tmp.extend(label[idx[cls]])
            
            if task > 0:
                np.random.shuffle(_tmp)
            
            shuffled_label.extend(_tmp)

        shuffled_label = np.asarray(shuffled_label)
        
        label = shuffled_label 

        idx = np.asarray([np.where(label==i) for i in np.unique(label)])
        
        if type == 'train':
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][(slot-1)*50:slot*50]
                    )
                )
        else:
            for ii in range(len(idx)):
                self.indeces.extend(
                    list(
                        np.roll(idx[ii],(shift-1)*100)[0][500:600]
                    )
                )
                
    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.dataset[self.indeces[index]]
 
 #jd's version to do rotation experiment on task 1 data
class GetAngleDataset(Dataset):

    def __init__(self, datatset_to_process, type='train', angle=0):
        super().__init__()
        self.dataset = datatset_to_process
        self.indeces1 = []
        self.indeces2 = []
        self.angle = angle
        self.type = type

        label = np.asarray([lbl for _,lbl in self.dataset])
        idx = np.asarray([np.where(label==i) for i in range(10)])
        #print(datatset_to_process[0], 'idx')
        if type == 'train':
            for ii in range(len(idx)):
                indxs = idx[ii][0][0:500]
                np.random.shuffle(indxs)
                
                self.indeces1.extend(
                    list(
                        indxs[0:250]
                    )
                )

                self.indeces2.extend(
                    list(
                        indxs[250:500]
                    )
                )

            #print(len(self.indeces1), len(self.indeces2), 'kutta')
        else:
            for ii in range(len(idx)):
                self.indeces1.extend(
                    list(
                        idx[ii][0][500:600]
                    )
                )

                self.indeces2.extend(
                    list(
                        idx[ii][0][500:600]
                    )
                )
        #print(self.indeces1 , self.indeces2, 'kutta2')
    def __len__(self):
        return len(self.indeces1)+len(self.indeces2)

    def __getitem__(self, index):
        
        if index >= 2500 and self.type == 'train':
            sample = self.dataset[self.indeces2[index-2500]]
            lbl = sample[1] + 10
            sample_ = torch.from_numpy(_image_aug(sample[0], self.angle)).type(torch.FloatTensor)
            #print(type(sample[1]))
        elif self.type == 'train':
            sample = self.dataset[self.indeces1[index]]
            lbl = sample[1]
            sample_ = sample[0]
        elif index>=1000 and self.type == 'test':
            #print(len(self.indeces2), index, 'hi')
            sample = self.dataset[self.indeces2[index-1000]]
            lbl = sample[1] + 10

            #print(lbl)
            sample_ = torch.from_numpy(_image_aug(sample[0], self.angle)).type(torch.FloatTensor)
        else:
            sample = self.dataset[self.indeces1[index]]
            lbl = sample[1]
            sample_ = sample[0]
        
        return (sample_, lbl)


class ReducedDataset(Dataset):
    '''To reduce a dataset, taking only samples corresponding to provided indeces.
    This is useful for splitting a dataset into a training and validation set.'''

    def __init__(self, original_dataset, indeces):
        super().__init__()
        self.dataset = original_dataset
        self.indeces = indeces

    def __len__(self):
        return len(self.indeces)

    def __getitem__(self, index):
        return self.dataset[self.indeces[index]]


class ReducedSubDataset(Dataset):
    '''To reduce & sub-sample a dataset, taking only those samples with label in [sub_labels] and at most [max] of them.

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None, max=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        counts = {}
        #print(sub_labels, 'sub_labels')
        for label in sub_labels:
            counts[label] = 0
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                if counts[label]<max:
                    counts[label] += 1
                    self.sub_indeces.append(index)
                
                    #print(counts, 'dgsfg')
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


class TransformedDataset(Dataset):
    '''To modify an existing dataset with a transform.
    This is useful for creating different permutations of MNIST without loading the data multiple times.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)


#----------------------------------------------------------------------------------------------------------#


def permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


#----------------------------------------------------------------------------------------------------------#


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Denormalize image, either single image (C,H,W) or image batch (N,C,H,W)"""
        batch = (len(tensor.size())==4)
        for t, m, s in zip(tensor.permute(1,0,2,3) if batch else tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor