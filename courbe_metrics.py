import os
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données du fichier CSV
log_data = pd.read_csv(os.path.join("checkpoints/training_log.csv"))

# Tracer la perte d'entraînement et de validation
plt.figure(figsize=(12, 6))
plt.plot(log_data["loss"], label="Training Loss")
plt.plot(log_data["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Enregistrer le graphique
plt.savefig(os.path.join("checkpoints", "training_validation_loss.png"))
plt.close()  # Fermer la figure pour libérer la mémoire

# Tracer la précision binaire pour la présence de voies
plt.figure(figsize=(12, 6))
plt.plot(log_data["fc_presence_binary_accuracy"], label="Training Binary Accuracy")
plt.plot(log_data["val_fc_presence_binary_accuracy"], label="Validation Binary Accuracy")
plt.title("Training and Validation Binary Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Binary Accuracy")
plt.legend()

# Enregistrer le graphique
plt.savefig(os.path.join("checkpoints", "training_validation_accuracy.png"))
plt.close()  # Fermer la figure pour libérer la mémoire