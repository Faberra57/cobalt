import os
import re
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# Chargement du modèle ONNX
onnx_model_path = 'cobalt.onnx'
session = ort.InferenceSession(onnx_model_path)

# Fonction pour prétraiter une image avant l'inférence (redimensionner à 128x128)
def preprocess_image(image_path):
    # Charger l'image .tif en niveaux de gris (1 canal)
    img = Image.open(image_path).convert("L")  # Convertir en niveaux de gris ('L' pour 1 canal)
    
    # Redimensionner l'image à 128x128
    img = img.resize((128, 128))  # Redimensionnement à 128x128
    
    # Convertir l'image en tableau numpy
    img_array = np.array(img).astype(np.float32)
    
    # Normalisation des pixels (mettre les valeurs entre 0 et 1)
    img_array = img_array / 255.0
    
    # Ajouter une dimension pour correspondre à (batch_size, channels, height, width)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension du batch (1, H, W)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension du canal (1, 1, H, W)
    
    return img_array


# Fonction pour faire une prédiction avec le modèle
def predict(image_path):
    # Prétraiter l'image
    img_array = preprocess_image(image_path)
    
    # Faire la prédiction (obtenir le nom de l'input du modèle)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    
    # Retourner la sortie
    return outputs

# Chemin vers les fichiers .tif
image_folder = 'dataset/archive/BBBC005_v1_images/BBBC005_v1_images'
tif_files = [f for f in os.listdir(image_folder) if f.endswith('.TIF')]

nb_images = 1000

# Vérifier qu'il y a suffisamment de fichiers .TIF
if len(tif_files) < nb_images:
    print(f"Il y a seulement {len(tif_files)} fichiers .TIF. Sélection de tous les fichiers.")
    tif_files = tif_files  # Sélectionner tous les fichiers si moins de nb_images
else:
    # Sélectionner nb_images fichiers aléatoires
    tif_files = random.sample(tif_files, nb_images)

print("nombre de fichier tif : ", len(tif_files))

# Listes pour stocker les valeurs réelles et les prédictions
all_actuals = []
all_predictions = []

# Test sur chaque fichier .tif
for tif_file in tif_files:
    image_path = os.path.join(image_folder, tif_file)
    
    # Utiliser une expression régulière pour extraire le nombre de cellules
    match = re.search(r'_C(\d+)_', tif_file)
    if match:
        nb_cells = int(match.group(1))  # Récupère le nombre comme entier
    else:
        print(f"Format incorrect pour {tif_file}, nombre de cellules non trouvé.")
        continue  # Ignorer les fichiers avec un format incorrect
    
    # Effectuer la prédiction pour l'image
    output = predict(image_path)
    
    # Ajouter les valeurs réelles et les prédictions à la liste
    all_actuals.append(nb_cells)
    all_predictions.append(output[0][0][0])  # Ajuster l'indexation si nécessaire

# Calcul des erreurs
errors = np.abs(np.array(all_predictions) - np.array(all_actuals))
mae = np.mean(errors)  # Erreur Absolue Moyenne
mse = np.mean(errors ** 2)  # Erreur Quadratique Moyenne
mape = np.mean(errors / np.array(all_actuals)) * 100  # Erreur Moyenne en Pourcentage

# Tracer la courbe des erreurs en valeur absolue
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Première sous-figure
plt.plot(sorted(all_actuals), sorted(errors), marker='o', linestyle='-', color='blue')
plt.title("Erreur moyenne en valeur absolue")
plt.xlabel("Nombre de cellules réel")
plt.ylabel("Erreur moyenne (valeurs absolues)")
plt.grid()

# Tracer le nuage de points (prédictions vs valeurs réelles)
plt.subplot(1, 2, 2)  # Deuxième sous-figure
plt.scatter(all_actuals, all_predictions, alpha=0.6, color='orange')
plt.plot(sorted(all_actuals), sorted(all_actuals), color='red', linestyle='--', label="Prédiction parfaite")  # Ligne y=x
plt.title("Nuage de points : Prédictions vs Réels")
plt.xlabel("Nombre de cellules réel")
plt.ylabel("Prédictions")
plt.legend()
plt.grid()

# Afficher les graphiques
plt.tight_layout()
plt.show()

# Afficher les métriques globales
print(f"MAE (Erreur Absolue Moyenne): {mae:.2f}")
print(f"MSE (Erreur Quadratique Moyenne): {mse:.2f}")
print(f"MAPE (Erreur Moyenne en Pourcentage): {mape:.2f}%")
