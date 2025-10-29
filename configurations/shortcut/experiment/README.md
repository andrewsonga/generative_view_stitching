## Reproducing Results

Here we provide the commands for reproducing all of the results from our paper. We run every experiment on a single NVIDIA H200 GPU (141GB VRAM) and report the resulting metrics. To run these experiments on a lower-VRAM GPU, such as the NVIDIA RTX A6000 (48GB VRAM), run with `@baseline/ours_scalable` and `algorithm=gvs_scalable_video_pose`, which requires less VRAM by denoising every context window one-by-one, and set `experiment.validation.batch_size=1`.

### 1. Comparisons with Baselines (Sections 4.1, B.3)

<details>
<summary><b>Straight Line</b></summary>

a) History-Guided Autoregressive Sampling
```
python main.py @baseline/autoregressive algorithm=dfot_video_pose dataset=straight_line @experiment/main_autoregressive_straight_line 
```

b) StochSync
```
python main.py @baseline/stochsync algorithm=stochsync_video_pose dataset=straight_line @experiment/main_stochsync_straight_line 
```

c) Ours
```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=straight_line @experiment/main_ours_straight_line 
```

</details>

<details>
<summary><b>Stairs</b></summary>

a) History-Guided Autoregressive Sampling
```
python main.py @baseline/autoregressive algorithm=dfot_video_pose dataset=stairs @experiment/main_autoregressive_stairs 
```

b) StochSync
```
python main.py @baseline/stochsync algorithm=stochsync_video_pose dataset=stairs @experiment/main_stochsync_stairs 
```

c) Ours
```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=stairs @experiment/main_ours_stairs
```
</details>

<details>
<summary><b>Panorama 1-loop</b></summary>

a) History-Guided Autoregressive Sampling
```
python main.py @baseline/autoregressive_rag algorithm=dfot_rag_video_pose dataset=panorama_1loop @experiment/main_autoregressive_panorama_1loop 
```

b) StochSync
```
python main.py @baseline/stochsync algorithm=stochsync_video_pose dataset=panorama_1loop @experiment/main_stochsync_panorama_1loop 
```

c) Ours
```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=panorama_1loop @experiment/main_ours_panorama_1loop
```

</details>

<details>
<summary><b>Circle 1-loop</b></summary>

a) History-Guided Autoregressive Sampling
```
python main.py @baseline/autoregressive_rag algorithm=dfot_rag_video_pose dataset=circle_1loop @experiment/main_autoregressive_circle_1loop 
```

b) StochSync
```
python main.py @baseline/stochsync algorithm=stochsync_video_pose dataset=circle_1loop @experiment/main_stochsync_circle_1loop 
```

c) Ours
```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=circle_1loop @experiment/main_ours_circle_1loop
```

</details>

<details>
<summary><b>Panorama 2-loop</b></summary>

a) History-Guided Autoregressive Sampling
```
python main.py @baseline/autoregressive_rag algorithm=dfot_rag_video_pose dataset=panorama_2loop @experiment/main_autoregressive_panorama_2loop 
```

b) StochSync
```
python main.py @baseline/stochsync algorithm=stochsync_video_pose dataset=panorama_2loop @experiment/main_stochsync_panorama_2loop 
```

c) Ours
```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=panorama_2loop @experiment/main_ours_panorama_2loop
```

</details>

<details>
<summary><b>Circle 2-loop</b></summary>

a) History-Guided Autoregressive Sampling
```
python main.py @baseline/autoregressive_rag algorithm=dfot_rag_video_pose dataset=circle_2loop @experiment/main_autoregressive_circle_2loop 
```

b) StochSync
```
python main.py @baseline/stochsync algorithm=stochsync_video_pose dataset=circle_2loop @experiment/main_stochsync_circle_2loop
```

c) Ours
```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=circle_2loop @experiment/main_ours_circle_2loop
```

</details>

<details>
<summary><b>Staircase Circuit</b></summary>

a) History-Guided Autoregressive Sampling
```
python main.py @baseline/autoregressive_rag algorithm=dfot_rag_video_pose dataset=staircase_circuit @experiment/main_autoregressive_staircase_circuit 
```

b) StochSync
```
python main.py @baseline/stochsync algorithm=stochsync_video_pose dataset=staircase_circuit @experiment/main_stochsync_staircase_circuit 
```

c) Ours
```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=staircase_circuit @experiment/main_ours_staircase_circuit
```

</details>

### 2. Applications (Sections 4.3, B.1)

<details>
<summary><b>Impossible Staircase</b></summary>

```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=impossible_staircase @experiment/application_impossible_staircase
```

</details>

<details>
<summary><b>Indefinite Staircase</b></summary>

```
python main.py @baseline/ours_scalable algorithm=gvs_scalable_video_pose dataset=indefinite_staircase_nframes1080_nloops9 @experiment/application_indefinite_staircase
```

</details>

### 3. Ablation on Omni Guidance and Stochasticity (Section 4.2)

<details>
<summary><b>Straight Line</b></summary>

a) No Omni Guidance, $\eta = 0$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0 algorithm=gvs_no_omniguide_video_pose dataset=straight_line @experiment/ablation_no_omniguide_eta0_straight_line
```

b) No Omni Guidance, $\eta = 0.5$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p5_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=straight_line @experiment/ablation_no_omniguide_eta0p5_maxstoch_straight_line
```

c) No Omni Guidance, $\eta = 0.9$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p9_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=straight_line @experiment/ablation_no_omniguide_eta0p9_maxstoch_straight_line
```

d) No Omni Guidance, $\eta = 1.0$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta1p0_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=straight_line @experiment/ablation_no_omniguide_eta1p0_maxstoch_straight_line
```

e) Omni Guidance, $\eta = 0$
```
python main.py @baseline/ours_jointgscale1p0_eta0 algorithm=gvs_video_pose dataset=straight_line @experiment/ablation_omniguide_eta0_straight_line
```

f) Omni Guidance, $\eta = 0.5$
```
python main.py @baseline/ours_jointgscale1p0_eta0p5_maxstoch algorithm=gvs_video_pose dataset=straight_line @experiment/ablation_omniguide_eta0p5_maxstoch_straight_line
```

g) Omni Guidance, $\eta = 0.9$
```
python main.py @baseline/ours_jointgscale1p0_eta0p9_maxstoch algorithm=gvs_video_pose dataset=straight_line @experiment/ablation_omniguide_eta0p9_maxstoch_straight_line
```

h) Omni Guidance, $\eta = 1.0$
```
python main.py @baseline/ours_jointgscale1p0_eta1p0_maxstoch algorithm=gvs_video_pose dataset=straight_line @experiment/ablation_omniguide_eta1p0_maxstoch_straight_line
```

</details>

<details>
<summary><b>Stairs</b></summary>

a) No Omni Guidance, $\eta = 0$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0 algorithm=gvs_no_omniguide_video_pose dataset=stairs @experiment/ablation_no_omniguide_eta0_stairs
```

b) No Omni Guidance, $\eta = 0.5$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p5_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=stairs @experiment/ablation_no_omniguide_eta0p5_maxstoch_stairs
```

c) No Omni Guidance, $\eta = 0.9$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p9_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=stairs @experiment/ablation_no_omniguide_eta0p9_maxstoch_stairs
```

d) No Omni Guidance, $\eta = 1.0$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta1p0_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=stairs @experiment/ablation_no_omniguide_eta1p0_maxstoch_stairs
```

e) Omni Guidance, $\eta = 0$
```
python main.py @baseline/ours_jointgscale1p0_eta0 algorithm=gvs_video_pose dataset=stairs @experiment/ablation_omniguide_eta0_stairs
```

f) Omni Guidance, $\eta = 0.5$
```
python main.py @baseline/ours_jointgscale1p0_eta0p5_maxstoch algorithm=gvs_video_pose dataset=stairs @experiment/ablation_omniguide_eta0p5_maxstoch_stairs
```

g) Omni Guidance, $\eta = 0.9$
```
python main.py @baseline/ours_jointgscale1p0_eta0p9_maxstoch algorithm=gvs_video_pose dataset=stairs @experiment/ablation_omniguide_eta0p9_maxstoch_stairs
```

h) Omni Guidance, $\eta = 1.0$
```
python main.py @baseline/ours_jointgscale1p0_eta1p0_maxstoch algorithm=gvs_video_pose dataset=stairs @experiment/ablation_omniguide_eta1p0_maxstoch_stairs
```

</details>

### 4. Ablation on Loop Closing and Omni Guidance (Sections 4.3, B.2)

<details>
<summary><b>Panorama 1-loop</b></summary>

a) No Loop Closing, No Omni Guidance
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p9_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=panorama_1loop @experiment/ablation_no_loopclose_no_omniguide_eta0p9_maxstoch_panorama_1loop
```

b) No Loop Closing, Omni Guidance
```
python main.py @baseline/ours_jointgscale1p0_eta0p9_maxstoch algorithm=gvs_video_pose dataset=panorama_1loop @experiment/ablation_no_loopclose_omniguide_eta0p9_maxstoch_panorama_1loop
```

c) Loop Closing, No Omni Guidance
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p9_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=panorama_1loop @experiment/ablation_loopclose_no_omniguide_eta0p9_maxstoch_panorama_1loop
```

d) Loop Closing, Omni Guidance
```
python main.py @baseline/ours_jointgscale1p0_eta0p9_maxstoch algorithm=gvs_video_pose dataset=panorama_1loop @experiment/ablation_loopclose_omniguide_eta0p9_maxstoch_panorama_1loop
```

</details>

<details>
<summary><b>Panorama 2-loop</b></summary>

a) No Loop Closing, No Omni Guidance
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p9_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=panorama_2loop @experiment/ablation_no_loopclose_no_omniguide_eta0p9_maxstoch_panorama_2loop
```

b) No Loop Closing, Omni Guidance
```
python main.py @baseline/ours_jointgscale1p0_eta0p9_maxstoch algorithm=gvs_video_pose dataset=panorama_2loop @experiment/ablation_no_loopclose_omniguide_eta0p9_maxstoch_panorama_2loop
```

c) Loop Closing, No Omni Guidance
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p9_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=panorama_2loop @experiment/ablation_loopclose_no_omniguide_eta0p9_maxstoch_panorama_2loop
```

d) Loop Closing, Omni Guidance
```
python main.py @baseline/ours_jointgscale1p0_eta0p9_maxstoch algorithm=gvs_video_pose dataset=panorama_2loop @experiment/ablation_loopclose_omniguide_eta0p9_maxstoch_panorama_2loop
```

e) Loop Closing, No Omni Guidance, $\eta = 0.8$
```
python main.py @baseline/ours_condgscale1p0_histgscale0p0_eta0p8_maxstoch algorithm=gvs_no_omniguide_video_pose dataset=panorama_2loop @experiment/ablation_loopclose_no_omniguide_eta0p8_maxstoch_panorama_2loop
```

f) Loop Closing, Omni Guidance, $\eta = 0.8$
```
python main.py @baseline/ours_jointgscale1p0_eta0p8_maxstoch algorithm=gvs_video_pose dataset=panorama_2loop @experiment/ablation_loopclose_omniguide_eta0p8_maxstoch_panorama_2loop
```

</details>

### 5. Limitations (Sections C.1, C.2)

<details>
<summary><b>External Image Conditioning</b></summary>

```
python main.py @baseline/ours algorithm=gvs_video_pose dataset=realestate10k_mini @experiment/limitation_ours_re10k_mini_context1
```

</details>

<details>
<summary><b>Loop Closing Wide-Baseline Viewpoints</b></summary>

```
# stitching multiple context windows (No Loop Closing)
python main.py @baseline/ours algorithm=gvs_video_pose dataset=forward_orbit_backward @experiment/limitation_ours_forward_orbit_backward_no_loopclose

# stitching multiple context windows (Loop Closing)
python main.py @baseline/ours algorithm=gvs_video_pose dataset=forward_orbit_backward @experiment/limitation_ours_forward_orbit_backward_loopclose

# diffusing a single context window
python main.py @baseline/autoregressive algorithm=dfot_video_pose dataset=forward_orbit_backward_loop_closing_window @experiment/limitation_fullsequence_diffusion_forward_orbit_backward_loop_closing_window
```

</details>