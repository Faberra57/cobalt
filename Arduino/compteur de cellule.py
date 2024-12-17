import os
import random
from PIL import Image
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import re

# Charger le modèle ONNX
def load_model(onnx_model_path):
    try:
        session = ort.InferenceSession(onnx_model_path)
        return session
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle ONNX : {e}")

# Prétraiter une image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
    image = np.array(image)  # Convertir en tableau NumPy

    if image is None:
        raise ValueError(f"Image non trouvée à l'emplacement : {image_path}")
    
    # Ajouter une dimension pour les canaux
    image = np.expand_dims(image, axis=0)  # De [H, W] à [1, H, W]
    image = image.astype(np.float32) / 255.0  # Normaliser
    
    # Ajouter la dimension du batch
    image_batch = np.expand_dims(image, axis=0)  # De [1, H, W] à [1, 1, H, W]
    
    return image_batch

# Effectuer une prédiction
def predict(session, image_path):
    input_name = session.get_inputs()[0].name
    
    img_array = preprocess_image(image_path)
    outputs = session.run(None, {input_name: img_array})
    
    return outputs[0][0].item()  # Extraire un scalaire avec .item()

# Traiter une seule image
def process_one_image(image_folder, model_path):
    tif_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.tif')]
    if not tif_files:
        raise FileNotFoundError("Aucun fichier .TIF trouvé dans le dossier.")
    
    # Sélectionner une image aléatoire
    selected_file = random.choice(tif_files)
    image_path = os.path.join(image_folder, selected_file)

    # Extraire le nombre réel à partir du nom du fichier
    matches = re.findall(r'_C(\d+)', selected_file)
    if not matches:
        raise ValueError(f"Format incorrect pour {selected_file}, aucun nombre trouvé après '_C'.")
    nb_cells = int(matches[-1])  # Utiliser la dernière correspondance

    # Charger le modèle et effectuer la prédiction
    session = load_model(model_path)
    prediction = predict(session, image_path)

    # Afficher les résultats dans le terminal
    print(f"Image : {selected_file}")
    print(f"Nombre réel : {nb_cells}")
    print(f"Prédiction : {int(prediction)}")

    # Afficher l'image avec la prédiction et la valeur réelle
    img = Image.open(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"Réel : {nb_cells} | Prédit : {int(prediction)}")
    plt.axis('off')
    #plt.show()

# Fonction principale
if __name__ == "__main__":
    image_folder = 'dataset/archive/BBBC005_v1_images/BBBC005_v1_images_resized_test'
    model_path = 'capsule_network_image_100_final.onnx'

    process_one_image(image_folder, model_path)
