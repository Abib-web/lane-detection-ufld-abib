import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import tensorflow as tf
from models.ufld_model import UFLDModel  # Assurez-vous que le chemin d'import est correct
from utils.config import Config  # Assurez-vous que le chemin d'import est correct
from datasets.tusimple_loader import TuSimpleDataset  # Importer la classe TuSimpleDataset

# Fonction pour afficher les prédictions
def display_predictions(image, lane_points, img_size=(224, 224)):
    """
    Affiche les prédictions (points de voie) sur l'image.
    """
    # Redimensionner l'image à la taille d'origine
    image = cv2.resize(image, img_size)
    
    # Dessiner les points de voie sur l'image
    for x, y in lane_points:  # Itérer directement sur les paires (x, y)
        if x > 0 and y > 0:  # Ignorer les points non détectés (valeurs 0)
            x = int(x)
            y = int(y)
            cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)  # Couleur bleue
    
    # Relier les points de voie par des lignes
    points = [(int(x), int(y)) for x, y in lane_points if x > 0 and y > 0]
    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], color=(0, 255, 0), thickness=2)  # Couleur verte
    
    # Afficher l'image
    cv2.imshow("Prédictions de voie", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def postprocess_predictions(predictions, threshold=0.1):  # Réduire le seuil pour plus de points
    # Filtrer les points de voie en fonction d'un seuil
    predictions[predictions < threshold] = 0
    return predictions

# Charger la configuration
config = Config()

# Chemin vers le modèle sauvegardé (format SavedModel)
model_path = os.path.join(config.checkpoints_dir, "ufld_model_final.keras")
print(model_path)

# Vérifier si le fichier de modèle existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier de modèle {model_path} n'existe pas.")

# Charger le modèle
print("Chargement du modèle...")
model = tf.keras.models.load_model(model_path)
print("Modèle chargé avec succès.")

# Créer le dataset de test
test_dataset = TuSimpleDataset(
    data_dir=os.path.join(config.data_dir, "TUSimple"),
    mode='test',  # Utiliser le mode 'test' pour les données de test
    batch_size=1,  # Batch size de 1 pour tester une seule image
    img_size=(780, 1200),
    shuffle=False
)

# Vérifier si le dataset de test est vide
if len(test_dataset) == 0:
    raise ValueError("Le dataset de test est vide.")

# Tester plusieurs images du dataset de test
for i in range(len(test_dataset)):
    image, label = test_dataset[i]  # Prendre l'image i du dataset

    # Afficher l'image originale pour vérification
    original_image = (image[0] * 255).astype(np.uint8)  # Convertir l'image normalisée en uint8
    cv2.imshow("Image originale", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Faire une prédiction
    print(f"Prédiction en cours pour l'image {i + 1}...")
    predictions = model.predict(image)
    print("Prédiction terminée.")

    # Extraire les prédictions pour les points de voie (fc_lanes)
    lane_predictions = predictions["fc_lanes"][0]  # Prendre la première prédiction du batch

    # Post-traitement des prédictions
    lane_predictions = postprocess_predictions(lane_predictions, threshold=0.1)  # Réduire le seuil

    # Dénormaliser les points de voie
    lane_points = lane_predictions.reshape(-1, 2)  # Redimensionner en (N, 2)
    lane_points[:, 0] *= 224  # Dénormaliser x (largeur de l'image)
    lane_points[:, 1] *= 224  # Dénormaliser y (hauteur de l'image)

    # Afficher les résultats
    display_predictions(original_image, lane_points)  # Afficher les prédictions

    # Enregistrer l'image avec les prédictions
    output_path = os.path.join(config.output_path_dir, f"prediction_{i}.png")
    cv2.imwrite(output_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    print(f"Résultat enregistré sous {output_path}")