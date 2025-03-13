import sys
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

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
        img_size=config.img_size,  
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

# Définir les fonctions de perte
losses = {
    "fc_lanes": tf.keras.losses.Huber(),  # Huber Loss
    "fc_presence": tf.keras.losses.BinaryCrossentropy()
}
loss_weights = {
    "fc_lanes": 1.0,
    "fc_presence": 0.5
}

# Compiler le modèle avec un taux d'apprentissage personnalisé
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
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
    save_format="tf",  # Sauvegarder en format TF
    monitor='val_loss',
    mode='min',
    verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir=os.path.join(config.checkpoints_dir2, "logs"))

# Entraîner le modèle
history = model.fit(
    train_dataset,
    epochs=config.epochs,
    validation_data=val_dataset,
    callbacks=[reduce_lr, checkpoint_callback, csv_logger]
)

# Sauvegarder le modèle final
model.save(os.path.join(config.checkpoints_dir2, "ufld_model_final"), save_format="tf")
model.summary()