"""
Convert the raw-data trained model to TFLite.
Uses CNN model which converts cleanly without SELECT_TF_OPS.
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = "best_har_raw_model.h5"
TFLITE_PATH = "har_model.tflite"

def convert():
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    # Print input/output shapes for verification
    print(f"Input shape:  {model.input_shape}")   # (None, 128, 3)
    print(f"Output shape: {model.output_shape}")  # (None, 6)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # For CNN model, we don't need SELECT_TF_OPS
    # If using LSTM, uncomment these:
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,
    #     tf.lite.OpsSet.SELECT_TF_OPS
    # ]
    # converter._experimental_lower_tensor_list_ops = False

    # Optimize for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Quantize to int8 for faster inference (slight accuracy loss)
    # converter.target_spec.supported_types = [tf.float16]

    print("Converting...")
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"✔ Saved: {TFLITE_PATH} ({len(tflite_model)/1024:.1f} KB)")
    
    # Verify the model works
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input:  {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
    
    # Test with random data
    test_input = np.random.randn(1, 128, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Test output: {output}")
    print(f"Predicted class: {np.argmax(output)}")
    
    print("\n✔ TFLite model verified successfully!")

if __name__ == "__main__":
    convert()