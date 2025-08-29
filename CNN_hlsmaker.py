# build_and_convert_student.py
# Python 3.9+ recommended

import os
import tensorflow as tf

# --- QKeras imports (ensure qkeras is installed) ---
from qkeras import QConv2D, QDense, QActivation
from qkeras.quantizers import quantized_bits

# --- hls4ml imports ---
import hls4ml
from hls4ml.converters import convert_from_keras_model
from hls4ml.utils import config_from_keras_model
from hls4ml.report import read_vivado_report

# ----------------------------
# 1) Build the inference-only Student model (no hint layers)
# ----------------------------

def build_student_inference(nbits=3, sym=0, kernel=(3, 3)):
    """
    Rebuild the Student network for inference, mirroring the Student layer
    names used during training so weights can be transferred by name.

    Shapes follow the 'valid' paddings of the Student model in the paper:
    Conv1a -> Conv1b -> Conv2a(6) -> Conv2b(6) -> Flatten(48) -> Dense10 -> Dense10 -> Dense2
    """
    inputs = tf.keras.Input(shape=(9, 16, 1), name='input_Student')

    # Block 1
    x = QConv2D(
        1, kernel,
        kernel_quantizer=quantized_bits(nbits, 0, sym),
        bias_quantizer=quantized_bits(nbits, 0, 1),
        padding='valid',
        name='Student_Conv1a'
    )(inputs)
    x = QActivation(f'quantized_relu({nbits})')(x)

    x = QConv2D(
        1, kernel,
        kernel_quantizer=quantized_bits(nbits, 0, sym),
        bias_quantizer=quantized_bits(nbits, 0, 1),
        padding='valid',
        name='Student_Conv1b'
    )(x)
    x = QActivation(f'quantized_relu({nbits})')(x)

    # Block 2
    x = QConv2D(
        6, kernel,
        kernel_quantizer=quantized_bits(nbits, 0, sym),
        bias_quantizer=quantized_bits(nbits, 0, 1),
        padding='valid',
        name='Student_Conv2a'
    )(x)
    x = QActivation(f'quantized_relu({nbits})')(x)

    x = QConv2D(
        6, kernel,
        kernel_quantizer=quantized_bits(nbits, 0, sym),
        bias_quantizer=quantized_bits(nbits, 0, 1),
        padding='valid',
        name='Student_Conv2b'
    )(x)
    x = QActivation(f'quantized_relu({nbits})')(x)

    # Head
    x = tf.keras.layers.Flatten(name='Student_Flatten')(x)  # expected 48 features
    x = QDense(
        10,
        kernel_quantizer=quantized_bits(nbits, 0, sym),
        bias_quantizer=quantized_bits(nbits, 0, 1),
        name='Student_Dense1'
    )(x)
    x = QActivation(f'quantized_relu({nbits})')(x)

    x = QDense(
        10,
        kernel_quantizer=quantized_bits(nbits, 0, sym),
        bias_quantizer=quantized_bits(nbits, 0, 1),
        name='Student_Dense2'
    )(x)
    x = QActivation(f'quantized_relu({nbits})')(x)

    # Final (keep as Dense FP32 if you trained that way; otherwise QDense works too)
    outputs = tf.keras.layers.Dense(2, name='Student_Output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Student_Inference')
    return model


# ----------------------------
# 2) (Optional) Transfer weights from your training-time model
# ----------------------------

def transfer_student_weights(student_infer, trained_model_path=None):
    """
    If you have a saved Keras model (with HintLossLayer), copy weights
    for Student_* layers by name. This avoids needing HintLossLayer at conversion.
    """
    if trained_model_path is None:
        print("[INFO] No trained model path provided; proceeding with random init.")
        return

    # If your training-time model uses custom objects (HintLossLayer), define a stub:
    class HintLossLayer(tf.keras.layers.Layer):
        def __init__(self, gamma=1.0, mname='Hint', **kwargs):
            super().__init__(**kwargs)
        def call(self, inputs, **kwargs):
            # Pass-through the first tensor (student branch) to preserve graph shape
            return inputs[0]

    custom_objects = {
        'QConv2D': QConv2D,
        'QDense': QDense,
        'QActivation': QActivation,
        'quantized_bits': quantized_bits,
        'HintLossLayer': HintLossLayer
    }

    print(f"[INFO] Loading trained model: {trained_model_path}")
    trained = tf.keras.models.load_model(trained_model_path, custom_objects=custom_objects, compile=False)

    copied, skipped = 0, 0
    for layer in student_infer.layers:
        if not layer.weights:
            continue
        try:
            src = trained.get_layer(layer.name)
            layer.set_weights(src.get_weights())
            copied += 1
        except Exception:
            skipped += 1
    print(f"[INFO] Weight transfer complete: copied={copied}, skipped={skipped}")


# ----------------------------
# 3) hls4ml conversion and Vivado synthesis
# ----------------------------

def convert_and_build_hls(model,
                          output_dir='hls4ml_prj_student',
                          part='xcvu13p-fhga2104-2L-e',
                          clock_period=2.38,
                          backend='Vitis',
                          reuse_factor=1):
    """
    Convert the QKeras model to hls4ml, generate HLS/VHDL, and run Vivado synthesis.
    """
    # Create hls4ml config. With QKeras, quantizers drive precision; we still set strategy/reuse.
    cfg = config_from_keras_model(model, granularity='name')
    cfg['Model'] = cfg.get('Model', {})
    cfg['Model']['Precision'] = cfg['Model'].get('Precision', 'ap_fixed<12,4>')  # not critical with QKeras
    cfg['Model']['ReuseFactor'] = reuse_factor
    cfg['Model']['Strategy'] = 'Latency'  # push for minimum latency
    #cfg['LayerName'] = cfg.get('LayerName', {})

    # Example: ensure fully streaming I/O (low-latency)
    cfg['Model']['IOType'] = 'io_stream'

    # Convs and Denses: force full unroll (ReuseFactor=1) and latency strategy
    for lname in [
        'Student_Conv1a', 'Student_Conv1b',
        'Student_Conv2a', 'Student_Conv2b',
        'Student_Dense1', 'Student_Dense2',
        'Student_Output'
    ]:
        cfg['LayerName'][lname]['Precision'] = 'ap_fixed<12,4>' #'ap_fixed<6,4>'

    print("[INFO] Converting model to hls4ml...")
    hls_model = convert_from_keras_model(
        model,
        hls_config=cfg,
        output_dir=output_dir,
        part=part,
        clock_period=clock_period,
        backend=backend
    )

    hls_model.write()

    print("[INFO] Building HLS project (csim off, synth+vsynth on)...")
    hls_model.build(csim=False, synth=True, vsynth=True)

    try:
        from hls4ml.utils import plot_model
        plot_model(hls_model, show_shapes=True)
    except Exception:
        pass

    print(f"\n[INFO] HLS/VHDL generated in: {os.path.abspath(output_dir)}")
    print("      HLS csynth report: myproject_prj/solution1/syn/report/myproject_csynth.rpt\n")

    return hls_model


if __name__ == '__main__':
    # ---- User knobs ----
    N_BITS = 4       # use 3 for the ultra-compact QStudent, or 4 for a slightly larger model
    SYM = 0          # symmetric quantization flag as in your training code
    KERNEL = (3, 3)
    TRAINED_MODEL = None  # e.g., 'student_with_hints.h5' if you want to transfer weights
    OUT_DIR = 'hls4ml_prj_student'
    PART = 'xcvu13p-fhga2104-2L-e'
    CLOCK = 2.38     # ns (≈ 420 MHz)
    BACKEND = 'Vitis'
    REUSE = 1        # 1 = maximum parallelism, minimum latency

    # 1) Build inference model
    student = build_student_inference(nbits=N_BITS, sym=SYM, kernel=KERNEL)
    student.summary()

    # 2) Optionally transfer weights from the training-time model
    #transfer_student_weights(student, TRAINED_MODEL)

    # 3) Convert to HLS/VHDL and run Vivado synthesis
    convert_and_build_hls(student,
                          output_dir=OUT_DIR,
                          part=PART,
                          clock_period=CLOCK,
                          backend=BACKEND,
                          reuse_factor=REUSE)
