# Self-supervised Learning for Music Genre Classification

### 1. Abstract
Self-supervised learning is in the spotlight as a way to solve the problem of data-hunger of deep learning. Especially, in the music domain, self-supervised learning is attracting solution as a way to solve the high-cost and high-labor problem of labeling numerous songs. However, self-supervised learning has a collapsing problem that outputs a constant solution. To solve this problem, a method has been devised to give the network a variety of augmentation that is difficult to match. Until now, a method of applying augmentation to the time axis or frequency axis of audio has been used, but we propose a method of using image augmentation by converting audio into a spectrogram. This method is easier, and has advantages in performance.

### 2. Dataset & Setting
(1) Download fma_small and fma_medium from the [LINK](https://github.com/mdeff/fma).  
(2) Put fma_small and fma_medium to ``./data`` folder.  
(3) Download ``ffmpeg.exe`` from [LINK](https://www.ffmpeg.org/download.html) and put it to ``./``.

### 3. Data Pre-processing
convert ``.wav`` files to ``.npy``

``python preprocessing.py --size small``  
``python preprocessing.py --size medium``  

### 4. Self-supervised Pre-training
- Arguments
  - num_epochs
  - batch_size
  - size : ``small`` or ``medium``
  - optim :  ``Adam`` or ``SGD``
  - backbone : ``resnet50`` or ``resnet101`` or ``resnet152``
  - aug : ``audio`` or ``image``
  - lam : hyper parameter for ``mixup`` augmentation
  - aug_p : probability for augmentation

- Example   
``python train_simsiam.py --num_epochs 100
                        --batch_size 64
                        --size small
                        --backbone resnet101
                        --aug image
                        --lam 0.7
                        --aug_p 0.6``

### 5. Transfer Learning to Downstream Task & Inference
- Arguments   
  - num_epochs
  - batch_size
  - size
  - aug
  - exp_path : ./expfolder
- Example  
``python transfer.py --num_epochs 100
                   --batch_size 64
                   --size small
                   --aug image
                   --exp_path 154874``

### 6. Final Report   
[Report](https://github.com/TaeMiKim/Musical-Applications-of-Machine-Learning/blob/main/project/Final_Report.pdf)

