import pandas as pd
import cv2
import os
folder_mask = ''
folder_image = ''

# important de lancer set_path_to_dataset avant de lancer les autres fonctions
def set_path_to_dataset(path_from_file_to_dataset):
    global folder_mask, folder_image
    folder_mask = os.path.join(path_from_file_to_dataset,'dataset/archive/BBBC005_v1_ground_truth/synthetic_2_ground_truth')
    folder_image = os.path.join(path_from_file_to_dataset,'dataset/archive/BBBC005_v1_images/BBBC005_v1_images')
#load mask
def load_data_mask(folder = folder_mask):
    global folder_mask, folder_image
    img_list = os.listdir(folder_mask)
    img_list.remove('.htaccess') # remove this file

    def get_num_cells(x):

        #SIMCEPImages_A13_C53_F1_s09_w2.TIF -> C53

        a = x.split('_') # e.g. ['SIMCEPImages', 'A13', 'C53', 'F1', 's09', 'w2.TIF']
        b = a[2] # e.g. C53
        num_cells = int(b[1:]) # e.g. 53
        
        return num_cells
    
    df = pd.DataFrame({'image_id': img_list})
    df['image_id'] = df[df['image_id'] != '.htaccess']
    df['nb_cells'] = df['image_id'].apply(get_num_cells)

    return df

#load image and mask

def load_data_image(folder=folder_image):
    global folder_mask, folder_image
    # folder = 'dataset/archive/BBBC005_v1_images'
    img_list = os.listdir(folder_image)
    mask_list = os.listdir(folder_mask)
    img_list.remove('.htaccess') # remove this file
    mask_list.remove('.htaccess') # remove this file

    def get_num_cells(x):
        #SIMCEPImages_A13_C53_F1_s09_w2.TIF -> C53

        a = x.split('_') # e.g. ['SIMCEPImages', 'A13', 'C53', 'F1', 's09', 'w2.TIF']
        b = a[2] # e.g. C53
        num_cells = int(b[1:]) # e.g. 53
        
        return num_cells
    
    def has_mask(x):
        if x in mask_list:
            return 'yes'
        else: return 'no'
    
    def get_blur(x):
        #SIMCEPImages_A13_C53_F1_s09_w2.TIF -> C53

        a = x.split('_') # e.g. ['SIMCEPImages', 'A13', 'C53', 'F1', 's09', 'w2.TIF']
        b = a[3] # e.g. 'F1
        blur = int(b[1:]) # e.g. 1

        return blur

    df = pd.DataFrame({'image_id': img_list})
    df['image_id'] = df[df['image_id'] != '.htaccess']
    df['nb_cells'] = df['image_id'].apply(get_num_cells)
    df['mask'] = df['image_id'].apply(has_mask)
    df['blur'] = df['image_id'].apply(get_blur)
    return df

#load image and ground truth 
# mask_or_image = 'mask' or 'image' 
def data_to_numpy(df,img_width,img_height,mask_or_image):
    global folder_mask, folder_image
    if mask_or_image == 'mask':
        folder = folder_mask
    else:
        folder = folder_image
    matrixs = []
    for image_id in df['image_id']:
        img =  img = cv2.imread(folder + '/' + image_id, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        matrixs.append(img)
    return matrixs

if __name__ == '__main__':
    folder = 'dataset/archive/BBBC005_v1_ground_truth/synthetic_2_ground_truth'
    df = load_data_mask(folder)
    print(df.head())
