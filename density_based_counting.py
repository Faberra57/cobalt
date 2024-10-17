

## Import des bibliothèques utiles 
import numpy as np 
import random as rd 
from matplotlib import pyplot as plt 
from sklearn.cluster import DBSCAN 

## Fonctions outil 
def _eligibility_to_post_conditionning(height  :int, 
                                       width : int, 
                                       image : object) -> bool : 
    '''
    IN : 
    ----
    height : hauteur voulue (en pixel) de l'image pour qu'elle puisse être éligible au post traitement (int)
    width : largeur voulue (en pixel) de l'image pour qu'elle puisse être éligible au post traitement (int)
    image : l'image en qui doit atre un array (object)
    
    OUT : 
    ----
    indice d'éligibilité : True si l'image peut être post traitée, False sinon (bool)
    '''


    ## Verification array numpy 
    if not isinstance(image, np.ndarray) : # Si ce n'est pas un array numpy
        print(r"L'image que vous voulez traiter n'est pas un array numpy (np.ndarray)") # On informe 
        return False # On retourne qu'on ne peut pas triater l'image
    
    if image.shape() == (height, width) : # Si elle a la bonne taille
        return True # On peut la traiter 
    
    else : 
        print(r"Les dimensions de l'image ne sont pas celles spécifiées pour l'éligibilité")
        return False 
    

def _DBSCAN_counting(image : np.ndarray) -> int : 
    '''
    IN : 
    ----
    image : image dont on veut compter les clusters (np.ndarray)

    OUT : 
    ---- 
    le nombre de clusters déterminés dans l'image (int) 
    '''

    # Pretraitement : normalisation 
    def image_normalise(image : np.ndarray) : 
        if image.max() == 255 : 
            return image/255.0 
    
    image = image_normalise(image=image) # On normalise l'image si ce n'est pas encore fait 

    pixels_entite_reperee = np.argwhere(image == 1) # On regarde les position des pixels blancs : ceux ou l'on aura repéré une entité 

    scan_DB = DBSCAN(eps=1.5, min_samples=2).fit(pixels_entite_reperee) # Algorithme dbscan 

    nombre_de_clusters = len(set(scan_DB.labels_)) - (1 if -1 in scan_DB.labels_ else 0) 

    return int(nombre_de_clusters) # On renvoie ne nombre de clusters 



## Fonction finale 
def nb_entity_definer(image : object,
                      height : int,
                      width : int) -> int : 
    '''
    IN : 
    ----
    image : l'image dont on veut compter les entités reconnues (object == np.ndarray pour fonctionner)
    height : hauteur voulue (en pixel) de l'image pour qu'elle puisse être éligible au post traitement (int)
    width : largeur voulue (en pixel) de l'image pour qu'elle puisse être éligible au post traitement (int)

    OUT : 
    ----
    nombre d'entités repérées (int)
    '''
    
    if _eligibility_to_post_conditionning(height=height, 
                                          width=width, 
                                          image=image) : 
        return _DBSCAN_counting(image=image)
    




