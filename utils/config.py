import os

class Config:
    def __init__(self):
        # Chemin vers le dossier des données
        self.data_dir = os.path.join("C:\\", "Users", "koneo", "Desktop", "lane-detection-ufld-abib", "data", "tusimple")
        
        # Paramètres d'entraînement
        self.batch_size = 8
        self.img_size = (448, 448)
        self.epochs = 15
        
        # Chemin vers le dossier des checkpoints
        self.checkpoints_dir = os.path.join("C:\\", "Users", "koneo", "Desktop", "lane-detection-ufld-abib", "checkpoints")
        self.checkpoints_dir1 = os.path.join("C:\\", "Users", "koneo", "Desktop", "lane-detection-ufld-abib", "checkpoints1")
        self.checkpoints_dir2 = os.path.join("C:\\", "Users", "koneo", "Desktop", "lane-detection-ufld-abib", "checkpoints2")
        self.output_path_dir = os.path.join("C:\\", "Users", "koneo", "Desktop", "lane-detection-ufld-abib", "outpout")

        # Créer le dossier des checkpoints s'il n'existe pas
        os.makedirs(self.checkpoints_dir, exist_ok=True)