import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "best_har_model.h5"
TFLITE_PATH = "har_model.tflite"

def convert():
    model = load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # IMPORTANT: allow Select TF ops (needed for LSTM TensorList ops)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Disable experimental lowering of tensor list ops (as error message suggests)
    converter._experimental_lower_tensor_list_ops = False

    # (Optional) mobile optimizations
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("✔ Saved", TFLITE_PATH)

if __name__ == "__main__":
    convert()
