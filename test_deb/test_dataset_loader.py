import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.tusimple_loader import TuSimpleDataset
import matplotlib.pyplot as plt

def visualize_batch(images, labels):
    """
    Visualise un batch d'images et leurs labels.
    """
    batch_size = images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))

    for i in range(batch_size):
        # Afficher l'image
        axes[i].imshow(images[i])
        axes[i].axis('off')

        # Afficher les labels
        lane_points = labels["fc_lanes"][i].reshape(-1, 2)  # Reshape en (N, 2)
        presence = labels["fc_presence"][i]

        # Dénormaliser les points de voie
        lane_points[:, 0] *= 224  # Dénormaliser x (largeur de l'image)
        lane_points[:, 1] *= 224  # Dénormaliser y (hauteur de l'image)

        # Dessiner les points de voie sur l'image
        for x, y in lane_points:
            if x > 0 and y > 0:  # Ignorer les points non détectés (valeurs 0)
                axes[i].scatter(x, y, c='red', s=10)  # Dessiner un point rouge

        # Afficher la présence des voies
        axes[i].set_title(f"Presence: {presence}")

    plt.show()

def test_tusimple_loader(data_dir, batch_size=100, img_size=(224, 224)):
    """
    Teste le TuSimpleDataset en chargeant un batch de données et en les visualisant.
    """
    # Créer le dataset
    dataset = TuSimpleDataset(
        data_dir=data_dir,
        mode='train',
        batch_size=batch_size,
        img_size=img_size,
        shuffle=True,
        validation_split=0.1
    )

    # Charger un batch de données
    images, labels = dataset[0]  # Prendre le premier batch

    # Afficher les informations sur le batch
    print(f"Nombre d'images dans le batch: {images.shape[0]}")
    print(f"Shape des images: {images.shape}")
    print(f"Shape des labels fc_lanes: {labels['fc_lanes'].shape}")
    print(f"Shape des labels fc_presence: {labels['fc_presence'].shape}")

    # Visualiser le batch
    visualize_batch(images, labels)

# Chemin vers le dossier contenant les données TuSimple
data_dir = r"C:\Users\Diaraye Barry\Desktop\ulfd-lane-detection\data\tusimple\TUSimple"

# Tester le TuSimpleDataset
test_tusimple_loader(data_dir, batch_size=4)