import tensorflow as tf
def check_gpu():
    # Enable memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
        return True
    else:
        print("No GPU found")
        return False