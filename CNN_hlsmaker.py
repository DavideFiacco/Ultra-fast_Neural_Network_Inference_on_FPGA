# build_and_convert_student.py
# Python 3.9+ recommended

import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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

def build_student_inference(nbits=4, sym=0, kernel=(3, 3)):
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
    print(" - cfg", cfg)
    cfg['Model'] = cfg.get('Model', {})
    cfg['Model']['Precision'] = cfg['Model'].get('Precision', 'ap_fixed<4,0>')  # not critical with QKeras
    cfg['Model']['ReuseFactor'] = reuse_factor
    cfg['Model']['Strategy'] = 'Latency'  # push for minimum latency

    cfg['Model']['IOType'] = 'io_stream'

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

# ----------------------------
# 4) EVALUATION: load weights & produce efficiency curve
# ----------------------------
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Sequence

# ---- MODEL & WEIGHTS ----
def build_quantized_student(nbits: int = 4, sym: int = 0, kernel=(3,3)):
    """Wrapper so eval code doesn't depend on the HLS knobs."""
    return build_student_inference(nbits=nbits, sym=sym, kernel=kernel)

def _load_training_savedmodel(savedmodel_dir: str):
    """
    Load the training-time SavedModel (which may include HintLossLayer, Teacher taps, etc.)
    with the minimum custom-objects needed to deserialize QKeras layers.
    """
    # Minimal stub for HintLossLayer used during training so Keras can restore the graph
    class HintLossLayer(tf.keras.layers.Layer):
        def __init__(self, gamma=1.0, mname='Hint', **kwargs):
            super().__init__(**kwargs)
        def call(self, inputs, **kwargs):
            # pass-through: first tensor is the Student branch
            return inputs[0]

    custom_objects = {
        'QConv2D': QConv2D,
        'QDense' : QDense,
        'QActivation': QActivation,
        'quantized_bits': quantized_bits,
        'HintLossLayer': HintLossLayer,
    }

    print(f"[INFO] Loading SavedModel from: {savedmodel_dir}")
    mdl = tf.keras.models.load_model(savedmodel_dir, custom_objects=custom_objects, compile=False)
    return mdl

def load_weights_from_savedmodel_into_student(student_infer: tf.keras.Model,
                                              savedmodel_dir: str,
                                              strict_names: bool = True) -> None:
    """
    Copy weights from a training-time SavedModel (with QKeras + hints) into the
    inference-only Student graph (same Student_* names). Prints a match summary.
    """
    trained = _load_training_savedmodel(savedmodel_dir)

    # Build a quick name -> layer dict for both sides
    tgt_by_name = {l.name: l for l in student_infer.layers if l.weights}
    src_by_name = {l.name: l for l in trained.layers       if l.weights}

    copied, skipped, mismatched = 0, 0, 0
    report = []

    # Helper: attempt relaxed name match if strict fails (e.g., prefixes/scopes)
    def find_src_layer(relaxed_name: str):
        # exact first
        if relaxed_name in src_by_name:
            return src_by_name[relaxed_name]
        # try endswith match (common when models are wrapped/scoped)
        candidates = [ln for ln in src_by_name if ln.endswith(relaxed_name)]
        if len(candidates) == 1:
            return src_by_name[candidates[0]]
        # try startswith (less common)
        candidates = [ln for ln in src_by_name if ln.startswith(relaxed_name)]
        if len(candidates) == 1:
            return src_by_name[candidates[0]]
        return None

    for tgt_name, tgt_layer in tgt_by_name.items():
        src_layer = src_by_name.get(tgt_name, None) if strict_names else find_src_layer(tgt_name)
        if src_layer is None:
            report.append(f"SKIP (not found): {tgt_name}")
            skipped += 1
            continue

        # Shape check (same number of weights and compatible shapes)
        try:
            src_w = src_layer.get_weights()
            tgt_w = tgt_layer.get_weights()
            if len(src_w) != len(tgt_w):
                report.append(f"MISMATCH (num vars): {tgt_name} src={len(src_w)} tgt={len(tgt_w)}")
                mismatched += 1
                continue
            ok = all(sw.shape == tw.shape for sw, tw in zip(src_w, tgt_w))
            if not ok:
                report.append(
                    f"MISMATCH (shape): {tgt_name} src={[s.shape for s in src_w]} "
                    f"tgt={[t.shape for t in tgt_w]}"
                )
                mismatched += 1
                continue

            tgt_layer.set_weights(src_w)
            copied += 1
            report.append(f"COPIED: {tgt_name}")
        except Exception as e:
            report.append(f"ERROR  : {tgt_name} -> {e}")
            mismatched += 1

    print("\n[INFO] Weight copy report:")
    for line in report:
        print("  ", line)
    print(f"[INFO] Summary: copied={copied}  skipped={skipped}  mismatched/errors={mismatched}")


def load_train_test_sector_npy(
    train_images_path: str,
    train_labels_path: str,
    test_images_path: str,
    test_labels_path: str,
    expect_shape=(9, 16),
    add_channel_axis=True
):
    """
    Loader:
      - allow_pickle=True (matches how files were saved)
      - labels[:, 0] = pT  |  labels[:, 1] = eta_small
      - returns:
          X_train, Y_train (both columns),
          X_test,  Y_test  (both columns),
          scaler fitted on Y_train
    """
    # Cope with accidental double extension on test images (*.npy.npy)
    if test_images_path.endswith('.npy.npy'):
        test_images_path = test_images_path[:-4]

    # --- load arrays exactly like the notebook ---
    Xtr = np.load(train_images_path, allow_pickle=True)
    Ytr = np.load(train_labels_path, allow_pickle=True)
    Xte = np.load(test_images_path,  allow_pickle=True)
    Yte = np.load(test_labels_path,  allow_pickle=True)

    # # Ensure numeric
    # if isinstance(Ytr, np.ndarray) and Ytr.dtype == object:
    #     Ytr = np.vstack(Ytr)
    # if isinstance(Yte, np.ndarray) and Yte.dtype == object:
    #     Yte = np.vstack(Yte)
    # Ytr = np.asarray(Ytr, dtype=np.float32)
    # Yte = np.asarray(Yte, dtype=np.float32)

    # # Shapes & channel axis
    # def _prep_images(X):
    #     if X.ndim == 3 and add_channel_axis:
    #         X = X[..., np.newaxis]  # (N, 9, 16, 1)
    #     if X.ndim != 4:
    #         raise ValueError(f"Unexpected image shape: {X.shape}")
    #     if X.shape[1:3] != expect_shape:
    #         raise ValueError(f"Expected (*,{expect_shape[0]},{expect_shape[1]},*), got {X.shape}")
    #     return X.astype(np.float32)

    # Xtr = _prep_images(Xtr)
    # Xte = _prep_images(Xte)

    # Fit MinMax scaler on BOTH targets (pT, eta_small) like in the notebook
    scaler = MinMaxScaler()
    scaler.fit(Ytr)

    print(" - train_images:", Xtr.shape, " train_labels:", Ytr.shape)
    print(" - test_images :", Xte.shape, " test_labels :", Yte.shape)

    return Xtr, Ytr, Xte, Yte, scaler


def default_pt_bins():
    """
    pt_bins = concat( linspace(2,10,9), linspace(12,20,5) )
              -> [2,3,4,5,6,7,8,9,10, 12,14,16,18,20]
    """
    left  = np.linspace(2., 10., 9)
    right = np.linspace(12., 20., 5)
    return np.concatenate((left, right))

def compute_turnon(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,        # columns: [pT, eta_small]
    scaler: MinMaxScaler,
    pt_bins: Optional[np.ndarray] = None,
    pt_threshold: float = 10.0,
    batch_size: int = 4096
):
    """
    - Predict 2 outputs with the Student
    - inverse_transform with the MinMaxScaler fitted on train_labels
    - Use true_pt = Y_test[:, 0]
    - Efficiency = N(true_pt in bin & pred_pt > threshold) / N(true_pt in bin)
    Returns x(bin centers) and eff (as in the notebook)
    """
    if pt_bins is None:
        pt_bins = default_notebook_pt_bins()

    print(" - Xtest, ", np.shape(X_test) )
    # Predict, then inverse-transform back to physics units
    y_pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0)
    print(" - ypredict, ", np.shape(y_pred_scaled) )
    # y_pred_scaled has shape (N, 2) in scaled space
    y_pred = scaler.inverse_transform(y_pred_scaled)

    true_pt = Y_test[:, 0].astype(np.float32)
    pred_pt = y_pred[:, 0].astype(np.float32)

    numer, _ = np.histogram(true_pt[pred_pt > pt_threshold], pt_bins)
    denum, _ = np.histogram(true_pt, pt_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        eff = numer / denum
        eff[np.isnan(eff)] = 0.0

    x = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    return x, eff, pt_bins


def plot_turnon(
    x: np.ndarray, eff: np.ndarray,
    th: float = 10.0,
    title: str = "QStudent 10 GeV eff curve",
    outfile: Optional[str] = None
):
    with plt.rc_context({'font.size': 13}):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, eff, s=20)
        ax.axvline(x=th, color='black', ls='--')
        ax.set_xlabel(r"$p_T$ [GeV/c]")
        ax.set_ylabel("efficiency")
        ax.set_ylim(-0.05, 0.90)
        ax.set_title(title)
        ax.grid(False)
        if outfile:
            fig.tight_layout()
            fig.savefig(outfile, dpi=200)
            print(f"[INFO] Saved: {outfile}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true',
                        help='Run efficiency curve evaluation instead of HLS synthesis.')
    parser.add_argument('--nbits', type=int, default=4)

    parser.add_argument('--train-images', type=str, help='Train_ImagesEta_9x16_plus_noise_SectorPhi1.npy')
    parser.add_argument('--train-labels', type=str, help='Train_LabelsEtaAndPt_9x16_plus_noise_SectorPhi1.npy')
    parser.add_argument('--test-images',  type=str, help='Test_ImagesEta_9x16_plus_noise_SectorPhi1.npy or .npy.npy')
    parser.add_argument('--test-labels',  type=str, help='Test_LabelsEtaAndPt_9x16_plus_noise_SectorPhi1.npy')

    parser.add_argument('--savedmodel', type=str,
                        help='Path to a training-time SavedModel dir (e.g., .../net.tf)')
    parser.add_argument('--strict-names', action='store_true',
                        help='Require exact layer-name matches when copying weights (default: relaxed).')


    parser.add_argument('--out', type=str, default='efficiency_curve.png')

    args, unknown = parser.parse_known_args()

    # in args.eval branch — load, copy weights, evaluate
    if args.eval:
        # Build student with your nbits
        model = build_quantized_student(nbits=args.nbits, sym=0, kernel=(3,3))

        # Load weights (SavedModel recommended for your case)
        if args.savedmodel:
            load_weights_from_savedmodel_into_student(model, args.savedmodel, strict_names=args.strict_names)
        else:
            raise SystemExit("Provide --savedmodel /path/to/net.tf")

        model.summary()

        for lyr in model.layers:
            if hasattr(lyr, "kernel_quantizer_internal"):
                print(lyr.name, type(lyr.kernel_quantizer_internal).__name__)

        # Load datasets like the notebook
        if not (args.train_images and args.train_labels and args.test_images and args.test_labels):
            raise SystemExit("Provide --train-images --train-labels --test-images --test-labels (notebook-style).")

        Xtr, Ytr, Xte, Yte, scaler = load_train_test_sector_npy(
            args.train_images, args.train_labels, args.test_images, args.test_labels
        )

        # Compute & plot
        pt_bins = default_pt_bins()
        x, eff, _ = compute_turnon(
            model=model, X_test=Xte, Y_test=Yte, scaler=scaler, pt_bins=pt_bins, pt_threshold=10.0
        )
        plot_turnon(x, eff, th=10.0, title=f"QStudent ({args.nbits}-bit) 10 GeV eff curve", outfile=args.out)

    else:
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

        # 2) Transfer weights from the training-time model
        if args.savedmodel:
            load_weights_from_savedmodel_into_student(student, args.savedmodel, strict_names=args.strict_names)
        else:
            raise SystemExit("Provide --savedmodel /path/to/net.tf")

        student.summary()

        # 3) Convert to HLS/VHDL and run Vivado synthesis
        convert_and_build_hls(student,
                            output_dir=OUT_DIR,
                            part=PART,
                            clock_period=CLOCK,
                            backend=BACKEND,
                            reuse_factor=REUSE)
