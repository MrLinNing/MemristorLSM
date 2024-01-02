## LSM for EEG and E-MNIST

EEG Dataset can be download from Willett, Francis et al. (2021), Data from: High-performance brain-to-text communication via handwriting, Dryad, Dataset, https://doi.org/10.5061/dryad.wh70rxwmv 

[Download link](https://datadryad.org/stash/dataset/doi:10.5061/dryad.wh70rxwmv).
We use the data in the folder `Datasets/t5.2019.05.08/`



### LSM-CLIP with supervised loss on first 26 classes by one-fc projection layer (22 classes for train, and 4 classes for zero-shot test)
```
bash run_lsm_clip_one_layer_26class_sup.sh
```

### Trainable SRNN-CLIP with supervised loss on first 26 classes by one-fc projection layer (22 classes for train, and 4 classes for zero-shot test)
```
bash sup_zero_shot_trainable_search.sh
```

### Trainable SRNN-Prototipical Nets with supervised loss on first 26 classes by one-fc projection layer (22 classes for train, and 4 classes for zero-shot test)
```
bash sup_proto_zero_shot.sh
```


