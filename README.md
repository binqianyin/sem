# Image Segmentation for Imbalance Digital Rock Images

This repository implements methods and frameworks for training neural network models to segment shale rock images. Compared with existing solutions, our method achieves a better IoU score for objects that are difficult to identify, providing a foundation for subsequent geological analysis. We plan to implement supervised learning, unsupervised learning, and reinforcement learning approaches. Currently, unsupervised learning and reinforcement learning modules are under development.

The supervised learning approach uses U-Net as a backbone. We enhanced U-Net through stack ensembling learning to mitigate variances in U-Net when identifying multiple objects in an image. We also adopted the focal dice loss to segment unbalanced objects.

## Requirements

- The following software were used:

  - PyTorch >= 1.10
  - tqdm
  - sklearn
  - CUDA >= 11.1

- Install
  
  ```bash
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    pip3 install tqdm sklearn
  ```

## Usage

For simplicity, we included a pretrained model in the repository.
Training full models requires an *A100* or *V100* GPU to hold all images in the GPU memory.
We will fine-tune our code in the future to eliminate this limitation.

Using the predict mode, you can find `0.png` under current directory, which is the predicted segmentation of the test image.

### Preparation

Before executing the code, we copy pretrained models and raw images to a directory outside of the repository.

```bash
git clone https://github.com/binqianyin/sem.git
mkdir data
cp -rf sem/data/* data/
```

### Using pretrained models (GPU is not required)

You can find `0.tif` under current directory, which is the predicted segmentation.

```bash
python preprocess.py --data-dir ../data/test --output-dir ../data/test/processed --stride 32 --size 128
python classification.py --mode test-multi --data-dir ../data/test/processed/ --model-dir ../data/ --device cpu
python classification.py --mode test-synthesizer --data-dir ../data/test/processed --model-dir ../data/ --device cpu
```

### Training models (GPU is required)

#### Preprocess Images

```bash
python preprocess.py --data-dir ../data/train --output-dir ../data/train/processed --stride 32 --size 128
python preprocess.py --data-dir ../data/test --output-dir ../data/test/processed --stride 32 --size 128
```

#### Raw U-Net

```bash
# Training U-Net
python classification.py --mode train --experiment multi --data-dir ../data/train/processed/ --model-dir ../data/
# Testing U-Net
python classification.py --mode test --experiment multi --data-dir ../data/test/processed/ --model-dir ../data/
# Predicting a test image using U-Net
python classification.py --mode test --experiment multi --data-dir ../data/test/ --model-dir ../data/ --output-dir ./
```

#### U-Net Ensembled

```bash
# Training individual U-Net models for each object
python classification.py --mode train --experiment backbone --data-dir ../data/train/processed/ --model-dir ../data/ --focal --alpha 0.9
# Training the ensemble model
python classification.py --mode train --experiment synthesizer --data-dir ../data/train/processed/ --model-dir ../data/ --focal
# Testing the ensemble model
python classification.py --mode test --experiment synthesizer --data-dir ../data/test/processed/ --model-dir ../data/
# Predicting a test image using the ensemble model
python classification.py --mode test --experiment synthesizer --data-dir ../data/test/ --model-dir ../data/ --output-dir ./
```

## Comparison with Linear Discriminant Analysis (LDA)

You are expected to see a much lower score (~0.22) than NN-based segmentation.

```bash
python lda.py --train-dir ../data/train --test-dir ../data/test
```
