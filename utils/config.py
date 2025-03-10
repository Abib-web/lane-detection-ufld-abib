import os

class Config:
    def __init__(self):
        # Chemin vers le dossier des données
        self.data_dir = os.path.join("C:\\", "Users", "Diaraye Barry", "Desktop", "ulfd-lane-detection", "data", "tusimple")
        
        # Paramètres d'entraînement
        self.batch_size = 8
        self.img_size = (288, 800)
        self.epochs = 10
        
        # Chemin vers le dossier des checkpoints
        self.checkpoints_dir = os.path.join("C:\\", "Users", "Diaraye Barry", "Desktop", "ulfd-lane-detection", "checkpoints")
        self.output_path_dir = os.path.join("C:\\", "Users", "Diaraye Barry", "Desktop", "ulfd-lane-detection", "outpout")

        # Créer le dossier des checkpoints s'il n'existe pas
        os.makedirs(self.checkpoints_dir, exist_ok=True)