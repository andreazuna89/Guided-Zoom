This is a repository containing the code used in

> [Sarah Adel Bargal*, Andrea Zunino*, Vitali Petsiuk, Jianming Zhang, Kate Saenko, Vittorio Murino, Stan Sclaroff. "Guided Zoom: Questioning Network Evidence for Fine-grained Classification". BMVC 2019 (oral)](https://bmvc2019.org/wp-content/uploads/papers/0061-paper.pdf)

__This software implementation is provided for academic research and non-commercial purposes only.  This implementation is provided without warranty.__

The caffe version is the implementation code of Guided Zoom using Excitation Backprop saliency method.
The pytorch version is the implementation code of Guided Zoom using GradCAM and RISE saliency methods.

## Prerequisites for Caffe version
1. The same prerequisites as Caffe
2. Excitation Backprop framework implemented in Caffe
3. Anaconda (python packages)

## Prerequisites for Pytorch version
1. Pytorch

## Quick Start for Caffe version
The provided repository contains the code for evidence pool generation, computing, saving and combining the conventional and evidence CNN softmax predictions. 

1) To generate the evidence pool use the code: evidence_pool_generation.py
2) To compute and save the softmax predicted by the conventional and evidence CNN use the code: save_softmax.py
3) For the final decision refinement use the code: decision_refinement.py

## Reference
```
@InProceedings{bargal2019guidedzoom,
author={Adel Bargal, Sarah and Zunino, Andrea and Petsiuk, Vitali and Zhang, Jianming and Saenko, Kate and Murino, Vittorio and Sclaroff, Stan},
  title={Guided Zoom: Questioning Network Evidence for Fine-grained Classification},
  booktitle={British Machine Vision Conference (BMVC)},
  month = {September},
  year={2019}
}
```
