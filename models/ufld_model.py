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
        self.backbone.trainable = trainable

        # Couche 1x1 pour ajuster la sortie de ResNet50
        self.conv1x1 = layers.Conv2D(256, (1, 1), activation='relu')

        # Global Average Pooling pour préserver les informations spatiales
        self.gap = layers.GlobalAveragePooling2D()

        # Couches fully connected supplémentaires
        self.fc1 = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.fc2 = layers.Dense(256, activation='relu')

        # Fully Connected pour la détection des lignes et leur présence
        self.fc_lanes = layers.Dense(4 * 56, activation=None, name="fc_lanes")  # Pas d'activation
        self.fc_presence = layers.Dense(4, activation='sigmoid', name="fc_presence")

    def call(self, inputs):
        x = self.backbone(inputs, training=self.backbone.trainable)
        x = self.conv1x1(x)
        x = self.gap(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        lanes = self.fc_lanes(x)
        presence = self.fc_presence(x)
        return {"fc_lanes": lanes, "fc_presence": presence} # Sortie sous forme de dictionnaire
    
    def get_config(self):
        config = super(UFLDModel, self).get_config()
        config.update({
            "trainable": self.backbone.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)