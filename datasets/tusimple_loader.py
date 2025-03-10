import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TuSimpleDataset(Sequence):
    def __init__(self, data_dir, mode='train', batch_size=8, img_size=(224, 224), shuffle=True, validation_split=0.1, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.validation_split = validation_split

        if self.mode == 'test':
            # Charger les données de test
            self.test_file = os.path.join(self.data_dir, "test_set", "test_tasks_0627.json")
            self.image_paths, self.labels = self.load_test_annotations()
        else:
            # Charger les données d'entraînement et de validation
            self.clips_dir = os.path.join(self.data_dir, "train_set", "clips")
            self.label_files = [
                os.path.join(self.data_dir, "train_set", "label_data_0313.json"),
                os.path.join(self.data_dir, "train_set", "label_data_0531.json"),
                os.path.join(self.data_dir, "train_set", "label_data_0601.json"),
            ]
            self.image_paths, self.labels = self.load_annotations()

            # Diviser les données en entraînement et validation
            if self.mode in ['train', 'val']:
                self.image_paths, self.val_image_paths, self.labels, self.val_labels = train_test_split(
                    self.image_paths, self.labels, test_size=self.validation_split, random_state=42
                )
                if self.mode == 'val':
                    self.image_paths = self.val_image_paths
                    self.labels = self.val_labels

        # Augmentation des données pour le mode train
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.on_epoch_end()

    def load_annotations(self):
        """Charge les annotations depuis les fichiers de labels"""
        image_paths = []
        labels = []

        for label_file in self.label_files:
            if not os.path.exists(label_file):
                print(f"Warning: Label file {label_file} not found.")
                continue

            with open(label_file, "r") as f:
                data = [json.loads(line) for line in f]

            for sample in data:
                raw_file = sample["raw_file"]
                if raw_file.startswith("clips/"):
                    raw_file = raw_file[len("clips/"):]
                img_path = os.path.join(self.clips_dir, raw_file)

                if not os.path.exists(img_path):
                    print(f"Warning: Image file {img_path} not found.")
                    continue

                lanes = sample["lanes"]
                h_samples = sample["h_samples"]
                labels.append((lanes, h_samples))
                image_paths.append(img_path)

        return image_paths, labels

    def load_test_annotations(self):
        """Charge les annotations pour les données de test"""
        image_paths = []
        labels = []

        if not os.path.exists(self.test_file):
            print(f"Warning: Test file {self.test_file} not found.")
            return image_paths, labels

        with open(self.test_file, "r") as f:
            data = [json.loads(line) for line in f]

        for sample in data:
            raw_file = sample["raw_file"]
            if raw_file.startswith("clips/"):
                raw_file = raw_file[len("clips/"):]
            img_path = os.path.join(self.data_dir, "test_set", "clips", raw_file)

            if not os.path.exists(img_path):
                print(f"Warning: Image file {img_path} not found.")
                continue

            lanes = sample["lanes"]
            h_samples = sample["h_samples"]
            labels.append((lanes, h_samples))
            image_paths.append(img_path)

        return image_paths, labels

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_img_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images, labels = self.__data_generation(batch_img_paths, batch_labels)
        return images, labels

    def __data_generation(self, batch_img_paths, batch_labels):
        images = []
        lanes_labels = []
        presence_labels = []

        for img_path, (lanes, h_samples) in zip(batch_img_paths, batch_labels):
            # Charger et prétraiter l'image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Unable to load image at {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size) / 255.0  # Normalisation

            # Appliquer l'augmentation des données pour le mode train
            if self.mode == 'train':
                image = self.datagen.random_transform(image)

            images.append(image)

            # Générer les labels pour les points de voie
            lane_points = []
            for lane in lanes:
                for x, y in zip(lane, h_samples):
                    if x > 0:
                        lane_points.append(x / 1280)  # Normalisation en x
                        lane_points.append(y / 720)   # Normalisation en y
                    else:
                        lane_points.append(0.0)  # Si pas détecté
                        lane_points.append(0.0)

            # Vérifier la taille et compléter si nécessaire
            max_size = 56  # Nombre fixe attendu
            if len(lane_points) < max_size:
                lane_points.extend([0.0] * (max_size - len(lane_points)))
            elif len(lane_points) > max_size:
                lane_points = lane_points[:max_size]  # Tronquer si trop grand

            lanes_labels.append(lane_points)

            # Générer les labels pour la présence de voies
            presence = [1.0 if any(x > 0 for x in lane) else 0.0 for lane in lanes]
            # S'assurer que la taille est de 4 (une valeur par voie)
            if len(presence) < 4:
                presence.extend([0.0] * (4 - len(presence)))
            elif len(presence) > 4:
                presence = presence[:4]

            presence_labels.append(presence)

        # Convertir en tableaux NumPy
        images = np.array(images, dtype=np.float32)
        lanes_labels = np.array(lanes_labels, dtype=np.float32)
        presence_labels = np.array(presence_labels, dtype=np.float32)

        return images, {
            "fc_lanes": lanes_labels,
            "fc_presence": presence_labels
        }

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]