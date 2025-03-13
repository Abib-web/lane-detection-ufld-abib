import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import tensorflow as tf
from models.ufld_model import UFLDModel
from utils.config import Config
from datasets.tusimple_loader import TuSimpleDataset

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

    # Afficher l'image
    cv2.imshow("Points de voie", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
model_path = os.path.join(config.checkpoints_dir1, "ufld_model_final")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier de modèle {model_path} n'existe pas.")
model = tf.keras.models.load_model(model_path)

# Créer le dataset de test
test_dataset = TuSimpleDataset(
    data_dir=os.path.join(config.data_dir, "TUSimple"),
    mode='test',
    batch_size=15,
    img_size=(780, 900),
    shuffle=False
)

# Vérifier si le dataset de test est vide
if len(test_dataset) == 0:
    raise ValueError("Le dataset de test est vide.")

# Tester plusieurs images du dataset de test
for i in range(len(test_dataset)):
    image, label = test_dataset[i]

    # Afficher l'image originale
    original_image = (image[0] * 255).astype(np.uint8)
    cv2.imshow("Image originale", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Redimensionner l'image pour l'inférence
    image_resized = cv2.resize(image[0], (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

    # Faire une prédiction
    predictions = model.predict(image_resized)
    
    lane_predictions = predictions['fc_lanes']
    # Post-traitement des prédictions
    lane_predictions = postprocess_predictions(lane_predictions, threshold=0.1)
    
    # Dénormaliser les points de voie
    lane_points = lane_predictions.reshape(4, -1, 2)  # Reshape en (4, N, 2)
    for j in range(4):
        lane_points[j][:, 0] *= original_image.shape[1]  # Mise à l'échelle en largeur
        lane_points[j][:, 1] *= original_image.shape[0]  # Mise à l'échelle en hauteur

    # Afficher uniquement les points de voie
    display_points_only(original_image.copy(), lane_points)

    # Enregistrer l'image avec les prédictions
    output_path = os.path.join(config.output_path_dir, f"prediction_{i}.png")
    cv2.imwrite(output_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    print(f"Résultat enregistré sous {output_path}")