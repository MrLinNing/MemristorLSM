## Resistive memory-based zero-shot liquid state machine for multimodal event data learning


The human brain is a complex spiking neural network (SNN) that learns multimodal signals in a zero-shot manner by generalizing existing knowledge. Remarkably, the brain achieves this with minimal power consumption, using event-based signals that propagate within its structure.  
However, mimicking the human brain in neuromorphic hardware presents both hardware and software challenges.
Hardware limitations, such as the slowdown of Moore's law and the von Neumann bottleneck, hinder the efficiency of digital computers. 
On the software side, SNNs are known for their difficult training, especially when learning multimodal signals.

To overcome these challenges, we propose a hardware-software co-design that combines a fixed and random liquid state machine (LSM) SNN encoder with trainable artificial neural network (ANN) projections. The LSM is physically implemented using analogue resistive memory, leveraging the inherent stochasticity of resistive switching to generate random weights. This highly efficient and nanoscale in-memory computing approach effectively addresses the von Neumann bottleneck and the slowdown of Moore's law. The ANN projections are implemented digitally, allowing for easy optimization using contrastive loss, which helps to overcome the difficulties associated with SNN training.

We experimentally implement this co-design on a 40 nm 256 Kb in-memory computing macro. We first demonstrate LSM-based event encoding through supervised classification and linear probing on the N-MNIST and N-TIDIGITS datasets. Based on that, we showcase the zero-shot learning of multimodal events, including visual and audio data association, as well as neural and visual data alignment for brain-machine interfaces. Our co-design achieves classification accuracy comparable to fully optimized software models. This not only results in a 152.83 and 393.07-fold reduction in training costs compared to state-of-the-art contrastive language-image pre-training (CLIP) and Prototypical networks, but also delivers a 23.34 and 161-fold improvement in energy efficiency compared to cutting-edge digital hardware, respectively.

These proof-of-principle prototypes not only demonstrate the efficient and compact neuromorphic hardware using in-memory computing, but also zero-shot learning multimodal events in a brain-inspired manner, paving the way for future edge neuromorphic intelligence.

### Dataset
For event-driven N-MNIST and N-TIDIGITS datasets, please download from [One drive](https://hkuhk-my.sharepoint.com/:f:/g/personal/linning_hku_hk/Epr0YCEKH-dLjfrwzErKlzsBU-LZRHsH0FmiOWxf6BIGJw?e=l4FeeJ).

For EEG dataset, please download from [One drive](https://hkuhk-my.sharepoint.com/:f:/g/personal/linning_hku_hk/Epr0YCEKH-dLjfrwzErKlzsBU-LZRHsH0FmiOWxf6BIGJw?e=l4FeeJ).

For the E-MNIST, the data will be downloaded from the torch dataloader.

The EEG and E-MNIST datasets have been processed to be event-driven data by data CLIP and RateEncoding methods, respectively. Please see the `data_process_enlarge.py` file in the EEG_clip/EEG_to_Image folder for details. 

### Hardware and Software Information
#### Hardware
```
GPU: NVIDIA GeForce RTX 3090 Ti
CPU: AMD Ryzen 9 7950X 16-Core Processor
Memory: 61G
```


#### Software
```
Python version:  3.9.12 (main, Apr  5 2022, 06:56:58) 
[GCC 7.5.0]
torch version:  1.13.0
numpy version:  1.21.5
matplotlib version:  3.5.1
sklearn version:  1.0.2
CUDA is available
CUDA version:  11.6
```




### LSM-based Supervised Classification on N-MNIST
```angular2html
cd LSM-NMNIST
```

### LSM-based Supervised Classification on N-TIDIGITS
```angular2html
cd LSM-NTIDIGITS
```

### LSM-based Contrastive Learning for N-MNIST and N-TIDIGITS 
```angular2html
cd LSM-CLIP-ATV
```

### LSM-based Contrastive Learning for EEG and E-MNIST
```angular2html
cd LSM-CLIP-BTV
```

