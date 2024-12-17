import os
from PIL import Image


def resize_images(input_folder, output_folder, size=(128, 128)):
    # Vérifie si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourt tous les fichiers du dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".tif"):
            # Chemin complet du fichier d'entrée
            input_path = os.path.join(input_folder, filename)
            # Chemin complet du fichier de sortie
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Ouvre l'image
                with Image.open(input_path) as img:
                    # Redimensionne l'image
                    resized_img = img.resize(size)
                    # Enregistre l'image redimensionnée
                    resized_img.save(output_path, format="TIFF")
                #print(f"Image {filename} redimensionnée et sauvegardée.")
            except Exception as e:
                print(f"Erreur avec l'image {filename}: {e}")

# Exemple d'utilisation
input_folder = "dataset/archive/BBBC005_v1_images/BBBC005_v1_images"
output_folder = "dataset/archive/BBBC005_v1_images/BBBC005_v1_images_resized"
resize_images(input_folder, output_folder)
