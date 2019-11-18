# Fashion Synthesis Benchmark

### 来源

Deepfashion 数据集下的一个分支：  [Fashion Synthesis](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html )



### 使用 Paper

- Be Your Own Prada: Fashion Synthesis with Structural Coherence

  [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Be_Your_Own_ICCV_2017_paper.pdf) [[web]](http://mmlab.ie.cuhk.edu.hk/projects/FashionGAN/) [[code(torch)]](https://github.com/zhusz/ICCV17-fashionGAN)   

- Language Guided Fashion Image Manipulation with Feature-wise Transformations

  [[paper](https://arxiv.org/pdf/1808.04000.pdf)]   

- Bilinear Representation for Language-based Image Editing Using Conditional Generative Adversarial Networks

  [[paper](https://arxiv.org/pdf/1903.07499.pdf)] [[code(pytorch)](https://github.com/vtddggg/BilinearGAN_for_LBIE)]  

  

### 代码目录

- get_label.py: 根据 Caption 获得图像标签（性别，颜色，袖长，款式）
- count_num.py: 分别统计性别，颜色，袖长，款式的图像数量， 基于 get_label.py