

# Ref-NeuS: Ambiguity-Reduced Neural Implicit Surface Learning for Multi-View Reconstruction with Reflection

## [Project Page](https://g3956.github.io/) |  [Paper](https://arxiv.org/pdf/2303.10840.pdf)

This is the official repo for the implementation of [Ref-NeuS: Ambiguity-Reduced Neural Implicit Surface Learning for Multi-View Reconstruction with Reflection](https://arxiv.org/pdf/2303.10840.pdf), Wenhang Ge, Tao Hu, Haoyu Zhao, Shu Liu, Ying-Cong Chen.

## Setup

Installation 

This code is built with pytorch 1.11.0. See ```requirements.txt``` for the python packages.

You can create an anaconda environment called refneus with the required dependencies by running:

```
conda create -n refneus python=3.7 
conda activate refneus  
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Data

Download data [ShinlyBlender](https://storage.googleapis.com/gresearch/refraw360/ref.zip).

Download the GT dense point cloud for evaluation from [Google Drive](https://drive.google.com/file/d/1HGTD3uQUr8WrzRYZBagrg75_rQJmAK6S/view?usp=sharing).

Make sure the data is organized as follows (we show an object helmet here):
<pre>
+-- ShinyBlender
|   +-- helmet
|       +-- test
|       +-- train
|       +-- dense_pcd.ply
|       +-- points_of_interest.ply
|       +-- test_info.json
|       +-- transforms_test.json
|       +-- transforms_train.json
    +-- toaster
</pre>

## Evaluation with pretrained model

Download the pretrained models [Pretrained Models for reconstruction evaluation](https://drive.google.com/file/d/17A0x04nyRc9QLd31R57tWz1tcn159vr2/view?usp=sharing), 
 [Pretrained Models for PSNR evaluation](https://drive.google.com/file/d/1wqFJBv3hAHbBTM49yQZ_Gctm2CV_QVrr/view?usp=sharing).

Run the evaluation script with

```python exp_runner.py --mode validate_mesh --conf ./confs/womask.conf --ckpt_path ckpt_path```

```ckpt_path``` is the path to the pretrained model. 

Make sure the ```data_dir``` in configuration file ```./confs/womask.conf``` points to the same object as pretrained model.

The output mesh will be in ```base_exp_dir/meshes```. You can specify the path ```base_exp_dir``` in the configuration file.

The evaluaton metrics will be written in ```base_exp_dir/result.txt```.

The error visulization are in ```base_exp_dir/vis_d2s.ply```. Points with large errors are marked in red.

We can also download our final meshes results [here](https://drive.google.com/file/d/1r1G4Lu3U2017PHgIImx7WXm_ERSfKaHv/view?usp=sharing). 

We also provide a function to make a video for surface normals and novel view synthesis. Run the evaluation script with

```python exp_runner.py --mode visualize_video --conf ./confs/womask.conf --ckpt_path ckpt_path```

The output videos will be in ```base_exp_dir/normals.mp4``` and ```base_exp_dir/video.mp4```.

## Train a model from scratch

We wiil release the training code soon.

## Citation


If you find our work useful in your research, please consider citing:

```
@article{ge2023ref,
  title={Ref-NeuS: Ambiguity-Reduced Neural Implicit Surface Learning for Multi-View Reconstruction with Reflection},
  author={Ge, Wenhang and Hu, Tao and Zhao, Haoyu and Liu, Shu and Chen, Ying-Cong},
  journal={arXiv preprint arXiv:2303.10840},
  year={2023}
}
```


## Acknowledgments

Our code is partially based on [NeuS](https://github.com/Totoro97/NeuS) project and some code snippets are borrowed from [NeuralWarp](https://github.com/fdarmon/NeuralWarp). Thanks for these great projects. 


