import sys
import os
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
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
        img_size=(224, 224),
        shuffle=True,
        validation_split=0.1
    )
    val_dataset = TuSimpleDataset(
        data_dir=os.path.join(config.data_dir, "TUSimple"),
        mode='val',
        batch_size=config.batch_size,
        img_size=(224, 224),
        shuffle=False
    )
except ValueError as e:
    print(f"Erreur lors de l'initialisation du dataset : {e}")
    sys.exit(1)

# Créer le modèle
model = UFLDModel(input_shape=(224, 224, 3))

# Définir les fonctions de perte
losses = {
    "fc_lanes": "mse",  # Perte pour les points de voie
    "fc_presence": "binary_crossentropy"  # Perte pour la présence de voies
}
loss_weights = {
    "fc_lanes": 1.0,
    "fc_presence": 0.5
}

# Compiler le modèle
model.compile(
    optimizer='adam',
    loss=losses,
    loss_weights=loss_weights,
    metrics={
        "fc_lanes": ["mae"],  # Métrique pour les points de voie
        "fc_presence": ["binary_accuracy"]  # Métrique pour la présence de voies
    }
)

# Callbacks
csv_logger = CSVLogger(os.path.join(config.checkpoints_dir, "training_log.csv"), append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(config.checkpoints_dir, "ufld_model_epoch_{epoch:02d}.keras"),
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Entraîner le modèle
model.fit(
    train_dataset,
    epochs=50,  # Augmenter le nombre d'epochs
    validation_data=val_dataset,
    callbacks=[reduce_lr, checkpoint_callback, csv_logger, early_stopping]
)

# Sauvegarder le modèle final
model.save(os.path.join(config.checkpoints_dir, "ufld_model_final.keras"))
model.summary()