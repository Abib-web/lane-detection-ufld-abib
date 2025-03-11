import sys
import os
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from datasets.tusimple_loader import TuSimpleDataset
from models.ufld_model import UFLDModel
from utils.config import Config

# Charger la configuration
config = Config()

# Créer les datasets
try:
    train_dataset = TuSimpleDataset(
        data_dir=os.path.join(config.data_dir, "TUSimple"),
        mode='train',
        batch_size=config.batch_size,
        img_size=config.img_size,  # Utiliser config.img_size
        shuffle=True,
        validation_split=0.1
    )
    val_dataset = TuSimpleDataset(
        data_dir=os.path.join(config.data_dir, "TUSimple"),
        mode='val',
        batch_size=config.batch_size,
        img_size=config.img_size,  # Utiliser config.img_size
        shuffle=False
)
except ValueError as e:
    print(f"Erreur lors de l'initialisation du dataset : {e}")
    sys.exit(1)

# Créer le modèle
model = UFLDModel(input_shape=(config.img_size[0], config.img_size[1], 3))

# Définir l'optimiseur avec un taux d'apprentissage initial de 0.001
optimizer = Adam(learning_rate=0.001)
# Définir les fonctions de perte
losses = {
    "fc_lanes": tf.keras.losses.MeanAbsoluteError(),  # Utiliser MAE au lieu de MSE
    "fc_presence": tf.keras.losses.BinaryCrossentropy()
}
loss_weights = {
    "fc_lanes": 1.0,
    "fc_presence": 0.5
}

# Compiler le modèle
model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights=loss_weights,
    metrics={
        "fc_lanes": ["mae"],  # Métrique pour les points de voie
        "fc_presence": ["binary_accuracy"]  # Métrique pour la présence de voies
    }
)

# Callbacks
csv_logger = CSVLogger(os.path.join(config.checkpoints_dir2, "training_log.csv"), append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(config.checkpoints_dir2, "ufld_model_epoch_{epoch:02d}"),
    save_best_only=True,
    save_format="tf",  # 🔹 Ajouter ce paramètre pour sauvegarder en format TF (au lieu de .keras)
    monitor='val_loss',
    mode='min',
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Nombre d'epochs à attendre avant d'arrêter
    restore_best_weights=True
)


# Entraîner le modèle
model.fit(
    train_dataset,
    epochs=config.epochs,
    validation_data=val_dataset,
    callbacks=[reduce_lr, checkpoint_callback, csv_logger, early_stopping]  # Ajoutez csv_logger ici
)

# Sauvegarder le modèle final
model.save(os.path.join(config.checkpoints_dir2, "ufld_model_final"), save_format="tf")
model.summary()
