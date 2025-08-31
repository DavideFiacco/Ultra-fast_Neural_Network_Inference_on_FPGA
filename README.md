# Ultra-fast_Neural_Network_Inference_on_FPGA

This repo contains a single Python driver, CNN_hlsmaker.py, that supports two workflows:

1) Evaluation mode (--eval)
Rebuild the Student QKeras model, load trained weights from a TensorFlow SavedModel, load train/test datasets (.npy), fit the same MinMaxScaler used in training, and produce the 10 GeV trigger turn-on (efficiency) curve.

2) HLS/VHDL generation (no --eval)
Rebuild the Student model for inference, convert it with hls4ml, and generate an HLS/VHDL project targeting an AMD/Xilinx FPGA (Vivado/Vitis HLS). Prints latency + resource estimates and writes reports/HDL.

### Dataset

Dataset available [here](https://www.dropbox.com/scl/fo/uoyhyo9tilsav4fbvyk94/AK03GkXakivgN8net7WJjt4?rlkey=cfn9jz2yhi1f1jmvnvwahk3bb&st=b6i7ijec&dl=0)

### Mode Evaluation

1) Rebuild Student (inference)
The script defines build_student_inference(...) which reconstructs the Student CNN.

2) Weight transfer
The helper load_weights_from_savedmodel_into_student(...) loads pretrained model.

3) Load Dataset
The function load_train_test_sector_npy(...) load training and testing set, estracting from the training the scaler values.

4) Compute turn on curve and plot
The functions compute_turnon(...) and plot_rutnon(...) respectively compute the binned efficiency and plot the efficiency curve of a 10 GeV trigger on the transverse momentum (pT) of the muon.

### Mode HLS implementation (No-evaluation)

CNN_hlsmaker.py – main script (QKeras model → hls4ml → HLS/VHDL + reports)

How the code works (high level)

1) Rebuild Student (inference)
The script defines build_student_inference(...) which reconstructs the Student CNN without training-time hint layers. Layer names (Student_Conv1a, Student_Conv1b, Student_Conv2a, Student_Conv2b, Student_Dense1, Student_Dense2, Student_Output) are preserved so you can copy weights by name if you have a trained model.

2) Weight transfer
The helper load_weights_from_savedmodel_into_student(...) loads pretrained model.

3) hls4ml conversion & synthesis
convert_and_build_hls(...):

4) Builds an hls4ml config with:

IOType='io_stream' (streaming I/O),

Strategy='Latency' (favor unrolling/pipelining),

per-layer ReuseFactor=1 (force full parallelism on Conv/Dense),

narrow result precision for accumulators (keeps add trees small to meet clock).

4) Converts the Keras/QKeras model to an HLS project targeting your FPGA part.

5) Writes the project and runs HLS synthesis (csynth) and Vivado OOC (vsynth).

6) Prints the report summary and saves a diagram (plot_model) of the HLS graph.

Where things are generated

Project root: hls4ml_prj_student/

VHDL: hls4ml_prj_student/firmware/hdl/

C-Synthesis report (latency/resources):
hls4ml_prj_student/myproject_prj/solution1/syn/report/myproject_csynth.rpt

Vivado OOC report (post-synth est.):
hls4ml_prj_student/myproject_prj/solution1/syn/report/ (same folder, multiple files)

### Command example

    python CNN_hlsmaker.py --eval \
    --nbits 4 \
    --savedmodel /path/to/QTeach10k_QStu1.4k_1_0_0_nbits4/net.tf \
    --train-images /path/to/dataset_cnn/Train_ImagesEta_9x16_plus_noise_SectorPhi1.npy \
    --train-labels /path/to/dataset_cnn/Train_LabelsEtaAndPt_9x16_plus_noise_SectorPhi1.npy \
    --test-images  /path/to/dataset_cnn/Test_ImagesEta_9x16_plus_noise_SectorPhi1.npy.npy \
    --test-labels  /path/to/dataset_cnn/Test_LabelsEtaAndPt_9x16_plus_noise_SectorPhi1.npy \
    --out turnon_qstudent4.png

Flags explained

--eval : switch to evaluation mode.

--nbits : quantization bits of the QStudent you want to instantiate (e.g., 3 or 4).

--savedmodel : path to the training-time SavedModel folder (contains saved_model.pb and variables/). The script loads it and copies Student_ layer weights* into the inference model.

--train-images, --train-labels : .npy files for training split (labels with two columns [pT, eta_small]).

--test-images, --test-labels : .npy files for test split (same structure).

--out : output PNG with the 10 GeV turn-on scatter.

### Requirements

- vivado and/or legacy vivado_hls and/or vitis_hls (actually this code has been run and tested with Vitis HLS - High-Level Synthesis v2024.1 (64-bit))
- conda, required to replicate environment
- ADD HLS4ML DEPENDENCIES (E.G. COMPILER)

#### Happy synthesizing!
