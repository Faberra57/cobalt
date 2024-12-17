import os
import random
from PIL import Image
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import cv2
import re

# Charger le modèle ONNX
def load_model(onnx_model_path):
    try:
        session = ort.InferenceSession(onnx_model_path)
        return session
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle ONNX : {e}")

# Prétraiter une image
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
    image = np.array(image)  # Convertir en tableau NumPy
    
    if image is None:
        raise ValueError(f"Image non trouvée à l'emplacement : {image_path}")
    
    # Redimensionner l'image
    batch_size, channels, height, width = input_shape
    if height != image.shape[0] or width != image.shape[1]:
        image = cv2.resize(image, (width, height))  # Redimensionner avec OpenCV
    
    # Ajouter une dimension pour les canaux
    image = np.expand_dims(image, axis=0)  # De [H, W] à [1, H, W]
    image = image.astype(np.float32) / 255.0  # Normaliser
    
    # Ajouter la dimension du batch
    image_batch = np.expand_dims(image, axis=0)  # De [1, H, W] à [1, 1, H, W]
    
    return image_batch

# Effectuer une prédiction
def predict(session, image_path):
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    img_array = preprocess_image(image_path, input_shape)
    outputs = session.run(None, {input_name: img_array})
    
    return outputs[0][0]  # Retourner la sortie prédite

# Traiter plusieurs fichiers et collecter les données pour l'évaluation
def process_and_evaluate(image_folder, nb_images, model_path, error_threshold=10):
    tif_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.tif')]
    if len(tif_files) < nb_images:
        print(f"Seulement {len(tif_files)} fichiers .TIF disponibles. Tous seront utilisés.")
        selected_files = tif_files
    else:
        selected_files = random.sample(tif_files, nb_images)
    
    print(f"Nombre de fichiers .TIF sélectionnés : {len(selected_files)}")
    
    all_actuals = []
    all_predictions = []
    total_error = 0
    total_squared_error = 0
    total_percentage_error = 0
    count = 0
    errors = {}

    session = load_model(model_path)

    for idx, file in enumerate(selected_files):
        image_path = os.path.join(image_folder, file)
        
        # Rechercher tous les nombres précédés par "_C"
        matches = re.findall(r'_C(\d+)', image_path)

        if matches:  
            if len(matches) == 1:  # Une seule correspondance
                nb_cells = int(matches[0])
            else:  # Plus d'une correspondance, prendre la dernière
                nb_cells = int(matches[-1])
            #print(f"Nombre de cellules détecté : {nb_cells}")
        else:
            raise ValueError(f"Format incorrect pour {image_path}, aucun nombre trouvé après '_C'.")
        
        try:
            # Effectuer une prédiction pour chaque image
            prediction = predict(session, image_path)


            
            # Collecter les prédictions et valeurs réelles
            all_actuals.append(nb_cells)
            all_predictions.append(prediction)
            
            # Calculer les erreurs
            error = abs(prediction - nb_cells)
            total_error += error
            total_squared_error += error**2
            if nb_cells > 0:
                percentage_error = (error / nb_cells) * 100
                total_percentage_error += percentage_error
            
            count += 1
            if nb_cells not in errors:
                errors[nb_cells] = []
            errors[nb_cells].append(error)
            
            # Afficher les images avec une grande erreur
            if error > error_threshold:
                print(f"Image : {file} | Réel : {nb_cells} | Prédit : {int(prediction)} | Erreur : {error}")
                img = Image.open(image_path)
                plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap='gray')
                plt.title(f"Erreur > {error_threshold}\nRéel : {nb_cells} | Prédit : {int(prediction)}")
                plt.axis('off')
                plt.show()
        
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {file} : {e}")
            continue
    
    # Calculer les métriques globales
    mae = total_error / count
    mse = total_squared_error / count
    mape = total_percentage_error / count

    # Calculer l'erreur moyenne pour chaque nombre de cellules
    mean_errors = {nb: np.mean(errs) for nb, errs in errors.items()}
    sorted_nb_cells = sorted(mean_errors.keys())
    sorted_mean_errors = [mean_errors[nb] for nb in sorted_nb_cells]

    # Tracer les graphiques
    plt.figure(figsize=(12, 6))
    
    # Graphique 1 : Erreur moyenne en fonction du nombre de cellules réels
    plt.subplot(1, 2, 1)
    plt.plot(sorted_nb_cells, sorted_mean_errors, marker='o', linestyle='-', color='blue')
    plt.title("Erreur moyenne en valeur absolue")
    plt.xlabel("Nombre de cellules réel")
    plt.ylabel("Erreur moyenne (valeurs absolues)")
    plt.grid()
    
    # Graphique 2 : Nuage de points (Prédictions vs Réels)
    plt.subplot(1, 2, 2)
    plt.scatter(all_actuals, all_predictions, alpha=0.6, color='orange')
    plt.plot(sorted_nb_cells, sorted_nb_cells, color='red', linestyle='--', label="Prédiction parfaite")
    plt.title("Nuage de points : Prédictions vs Réels")
    plt.xlabel("Nombre de cellules réel")
    plt.ylabel("Prédictions")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

    # S'assurer que les métriques sont des scalaires
    mae = np.mean(mae) if isinstance(mae, np.ndarray) else mae
    mse = np.mean(mse) if isinstance(mse, np.ndarray) else mse
    mape = np.mean(mape) if isinstance(mape, np.ndarray) else mape

    # Afficher les métriques globales avec les bons formats
    print(f"MAE (Erreur Absolue Moyenne): {mae:.2f}")
    print(f"MSE (Erreur Quadratique Moyenne): {mse:.2f}")
    print(f"MAPE (Erreur Moyenne en Pourcentage): {mape:.2f}%")


# Fonction principale
if __name__ == "__main__":
    image_folder = 'dataset/archive/BBBC005_v1_images/BBBC005_v1_images_resized_test'

    model_path = 'model/cnn.onnx'
    #model_path = 'CapsuleNetworks/capsule_network.onnx'
    nb_images = 500  # Nombre d'images à traiter
    
    process_and_evaluate(image_folder, nb_images, model_path)


