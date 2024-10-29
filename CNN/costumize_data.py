import torch
import cv2  # Make sure you have OpenCV installed 
import os
class CustomDataset(torch.utils.data.Dataset):
    global folder_mask, folder_image    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.folder_mask =''
        self.folder_image = ''
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_id']
        label = (self.dataframe.iloc[idx]['nb_cells'])
        # Load the image using OpenCV
        image = cv2.imread(self.folder_image + '/' + img_path, cv2.IMREAD_GRAYSCALE)  # OpenCV loads images in BGR format
        # Convert the image to PIL format
        
        if self.transform:
            image = self.transform(image)
        #print('from costumise', ( image ))
        return image , label
    def set_path_to_dataset(self, path_from_file_to_dataset):
        self.folder_mask = os.path.join(path_from_file_to_dataset,'dataset/archive/BBBC005_v1_ground_truth/synthetic_2_ground_truth')
        self.folder_image = os.path.join(path_from_file_to_dataset,'dataset/archive/BBBC005_v1_images/BBBC005_v1_images')
