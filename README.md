# Targeted Transferable Attack against Deep Hashing Retrieval
This is the code for "[Targeted Transferable Attack against Deep Hashing Retrieval]". 


## Requirements
- python 3.8 
- torch 1.10
- torchvision 0.11
- numpy 1.20

## Implementation

#### Overview of previous methods
- We consider four previous targeted attack methods as our competitors: [DHTA (P2P)](https://github.com/jiawangbai/DHTA-master), [THA](https://github.com/xunguangwang/Targeted-Attack-and-Defense-for-Deep-Hashing), [ProS_GAN](https://github.com/xunguangwang/ProS-GAN) and [NAG](https://github.com/SugarRuy/CVPR21_Transferred_Hash). 
- To evaluate the targeted transferability of different methods, we test the performance on five deep hashing methods, i.e., [DPSH](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf), [HashNet](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf), [CSQ](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf), [DSDH](https://papers.nips.cc/paper/6842-deep-supervised-discrete-hashing.pdf) and [ADSH](https://cs.nju.edu.cn/lwj/paper/AAAI18_ADSH.pdf). Our implementations are modified based on [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch). 


#### Overview of dirs and files
- `data` contains the dataset files utilized in this paper.
- `Hash` contains the implementations of five deep hashing methods.
- `utils` contains all the tools used for training models.
- `models` contains the implementations of our method.


## Usage
#### Train deep hashing models
You can easily train deep hashing models by replacing the path of data in the code, and then run
```
cd Hash
python DPSH.py
python HashNet.py
python CSQ.py
python DSDH.py
python ADSH.py
``` 

#### Generate anchor code
After setting the dataset and target model paths, you can generate anchor code by running 
```
cd models
python IAO.py 
```

#### Attack by our TTA-GAN method
Initialize the hyperparameters following our paper and then run 
```
cd models
python TTA_GAN.py 
```

#### Ensemble attack
To conduct model ensemble attack, you can run
```
cd models
python Ens.py 
```


## Acknowledgement
The codes are modified based on [Wang et al. 2021](https://github.com/xunguangwang/ProS-GAN).


## Cite
If you find this work is useful, please cite the following:
```
@inproceedings{zhu2023targeted,
  title={Targeted Transferable Attack against Deep Hashing Retrieval},
  author={Zhu, Fei and Zhang, Wanqian and Wu, Dayan and Wang, Lin and Li, Bo and Wang, Weiping},
  booktitle={Proceedings of the 5th ACM International Conference on Multimedia in Asia},
  pages={1--7},
  year={2023}
}
```