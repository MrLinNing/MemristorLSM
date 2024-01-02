## Contrastive Learning on LSM-based RRAM for Zero-shot Classification


### Baselines: zero-shot simulation rram hardware or trainable SRNN
```angular2html
bash run_zero_shot_sim_pd64.sh
bash run_zero_shot_trainable.sh
```

### Baselines: zero-shot simulation for LSM-based or SRNN-based prototype Network
```angular2html
bash run_prototype.sh
bash run_prototype_trainable.sh
```

### Run Our Model on RRAM platform
```angular2html
bash run_zero_shot_rram.sh
```

### Accuracy Comparisions of Audio Search Image 

| Model              | 1-7 Accuracy(%) | 8-9 Accuracy (%) |Projection Dimension|
|--------------------|-------------|---------------|----------------|
| SRNN-based CLIP (trainable)              | 82.28     |  74.0      |64|
| SRNN-based PrototypeNet (trainable)      | 50.25      | 82.14           |64|
| SRNN-based PrototypeNet (trainable)      | 53     | 76.11         |128|
| SRNN-based PrototypeNet (trainable)      | 54.5     | 83.48       |256|
| SRNN-based PrototypeNet (trainable)      | 60.2    | 80.58      |512|
| SRNN-based PrototypeNet (trainable)      | 52.9   | 82.36     |1024|
| LSM-based PrototypeNet  | 50.25 | 81.69           | 64            |
| LSM-based PrototypeNet  | 50.00 | 77.9          | 128           |
| LSM-based PrototypeNet  | 50.6 | 69.41         | 256           |
| LSM-based PrototypeNet  | 51.2 | 73.66         | 512           |
| LSM-based PrototypeNet  | 56.9 | 68.9         | 1024          |
| **Our (Software)** |  **66.1**  |  **88**           |64|
| **Our (Software)** |  **67.4**  |  **88.5**           |128|
| **Our (RRAM)**     | **67.42**   |  **87.5**           |64|


