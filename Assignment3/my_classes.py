import torch

class Dataset(torch.utils.data.Dataset):
  #'Characterizes a dataset for PyTorch'
  def __init__(self, image_paths):
        #'Initialization'
        self.image_paths = image_paths

  def __len__(self):
        #'Denotes the total number of samples'
        return len(self.image_paths)

  def __getitem__(self, index):
        # Select sample
        image_path = self.image_paths[index]
        return image_path
