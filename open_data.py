import pandas as pd
import os

def get_data(folder):
    # folder = 'dataset/archive/BBBC005_v1_ground_truth/synthetic_2_ground_truth'
    img_list = os.listdir(folder)
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

if __name__ == '__main__':
    folder = 'dataset/archive/BBBC005_v1_ground_truth/synthetic_2_ground_truth'
    df = get_data(folder)
    print(df.head())
