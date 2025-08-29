# Ultra-fast_Neural_Network_Inference_on_FPGA

This repository contains a single Python script that:

rebuilds the Student inference-only CNN in QKeras (no hint-loss layers),

converts it to an hls4ml project targeting an AMD/Xilinx FPGA,

generates C++ HLS + VHDL,

launches HLS synthesis and out-of-context (OOC) Vivado synthesis, and

prints latency and resource reports and points you to the report files.

It implements the low-bit (3–4 bit) QStudent design described in the paper and pushes for minimum latency (full unroll / reuse factor 1, streaming I/O, “Latency” strategy).

### File layout

CNN_hlsmaker.py – main script (QKeras model → hls4ml → HLS/VHDL + reports)

How the code works (high level)

1) Rebuild Student (inference)
The script defines build_student_inference(...) which reconstructs the Student CNN without training-time hint layers. Layer names (Student_Conv1a, Student_Conv1b, Student_Conv2a, Student_Conv2b, Student_Dense1, Student_Dense2, Student_Output) are preserved so you can copy weights by name if you have a trained model.

2) (Optional) Weight transfer
If you trained with custom HintLossLayer, the helper transfer_student_weights(...) loads your model with a stubbed loss layer and copies weights by layer name to the inference graph.

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

### Requirements

- vivado and/or legacy vivado_hls and/or vitis_hls (actually this code has been run and tested with Vitis HLS - High-Level Synthesis v2024.1 (64-bit))
- Python 3.9+ recommended

- - Packages:

- - tensorflow==2.12.* (or a TF2.x known to work with your QKeras/hls4ml combo)

- - qkeras==0.9.*

- - hls4ml==0.8.*


#### Happy synthesizing!