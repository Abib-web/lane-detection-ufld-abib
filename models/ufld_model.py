import tensorflow as tf
from tensorflow.keras import layers

class UFLDModel(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), trainable=True, **kwargs):
        super(UFLDModel, self).__init__(**kwargs)

        # Backbone : ResNet50
        self.backbone = tf.keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        self.backbone.trainable = trainable  # Permet d'activer/désactiver l'apprentissage des poids du backbone

        # Couche 1x1 pour ajuster la sortie de ResNet50
        self.conv1x1 = layers.Conv2D(256, (1, 1), activation='relu')
        
        # Pooling pour transformer la sortie en un vecteur
        self.global_pool = layers.GlobalAveragePooling2D()

        # Fully Connected pour la détection des lignes et leur présence
        self.fc_lanes = layers.Dense(56, activation='sigmoid', name="fc_lanes")  # 56 points pour la prédiction des lignes
        self.fc_presence = layers.Dense(4, activation='sigmoid', name="fc_presence")  # 4 valeurs pour indiquer la présence des lignes

    def call(self, inputs):
        x = self.backbone(inputs, training=self.backbone.trainable)  # Assure que le backbone respecte `trainable`
        x = self.conv1x1(x)  # Réduction des canaux
        x = self.global_pool(x)  # Compression en vecteur
        lanes = self.fc_lanes(x)  # Prédiction des points de ligne
        presence = self.fc_presence(x)  # Prédiction de la présence des lignes
        return {"fc_lanes": lanes, "fc_presence": presence}  # Sortie sous forme de dictionnaire

    def get_config(self):
        config = super(UFLDModel, self).get_config()
        config.update({
            "trainable": self.backbone.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
