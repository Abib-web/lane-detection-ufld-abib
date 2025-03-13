import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import tensorflow as tf
from models.ufld_model import UFLDModel
from utils.config import Config

# Fonction pour afficher uniquement les points de prédiction
def display_points_only(image, lane_points):
    """
    Affiche les points de voie sur l'image sans les relier par des lignes.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Bleu, Vert, Rouge, Jaune

    # Vérifier si on a bien 4 ensembles de points
    if len(lane_points) != 4:
        print("Attention : le modèle ne fournit pas exactement 4 lignes.")
        return

    # Tracer chaque ensemble de points avec une couleur différente
    for i, points in enumerate(lane_points):
        points = [(int(x), int(y)) for x, y in points if x > 0 and y > 0]  # Filtrer les points invalides
        for point in points:
            cv2.circle(image, point, radius=5, color=colors[i], thickness=-1)  # Tracer un cercle pour chaque point

    return image

# Fonction pour post-traiter les prédictions
def postprocess_predictions(predictions, threshold=0.2):
    """
    Filtre les prédictions en dessous d'un certain seuil.
    """
    predictions[predictions < threshold] = 0
    return predictions

# Charger la configuration
config = Config()

# Charger le modèle
model_path = os.path.join(config.checkpoints_dir, "ufld_model_final")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier de modèle {model_path} n'existe pas.")
model = tf.keras.models.load_model(model_path)

# Chemin vers la vidéo
video_path = "C:/Users/koneo/Downloads/3398942-uhd_2160_3840_30fps.mp4"

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Impossible d'ouvrir la vidéo à l'emplacement : {video_path}")

# Boucle pour traiter chaque frame de la vidéo
while True:
    ret, frame = cap.read()
    if not ret:
        break  # Sortir de la boucle si la vidéo est terminée

    # Redimensionner l'image pour l'inférence (garder une taille raisonnable)
    original_height, original_width = frame.shape[:2]
    image_resized = cv2.resize(frame, (640, 360))  # Redimensionner à une taille plus grande (640x360)
    resized_height, resized_width = image_resized.shape[:2]

    # Préparer l'image pour le modèle
    image_for_model = cv2.resize(image_resized, (224, 224))  # Redimensionner à la taille attendue par le modèle
    image_for_model = np.expand_dims(image_for_model, axis=0).astype(np.float32) / 255.0

    # Faire une prédiction
    predictions = model.predict(image_for_model)
    lane_predictions = predictions['fc_lanes']

    # Post-traitement des prédictions
    lane_predictions = postprocess_predictions(lane_predictions, threshold=0.2)

    # Dénormaliser les points de voie
    lane_points = lane_predictions.reshape(4, -1, 2)  # Reshape en (4, N, 2)
    for j in range(4):
        lane_points[j][:, 0] *= resized_width  # Mise à l'échelle en largeur
        lane_points[j][:, 1] *= resized_height  # Mise à l'échelle en hauteur

    # Afficher les points de voie sur la frame redimensionnée
    frame_with_points = display_points_only(image_resized.copy(), lane_points)

    # Redimensionner l'affichage final pour mieux voir la route
    display_frame = cv2.resize(frame_with_points, (1080, 720))  # Afficher en 1280x720

    # Afficher la frame avec les points de voie
    cv2.imshow("Prédictions de voie", display_frame)

    # Quitter si l'utilisateur appuie sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()