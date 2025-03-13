import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
gpu_info = tf.config.experimental.get_memory_info('GPU:0')
print(f"VRAM utilisée: {gpu_info['current']} MB / {gpu_info['peak']} MB")
print("GPU disponible :", tf.config.list_physical_devices('GPU'))