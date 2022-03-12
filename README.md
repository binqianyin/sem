# A Segmentation Pipeline for Digital Rock Images

```[bash]
# preprocess
python preprocess.py --data-dir ../data/train --output-dir ../data/train/processed --stride 32 --size 128
python preprocess.py --data-dir ../data/test --output-dir ../data/test/processed --stride 32 --size 128
python preprocess.py --data-dir ../data --output-dir ../data/ --stride 32 --size 128 --hybrid
python preprocess.py --data-dir ../data/fine-tune --output-dir ../data/fine-tune/processed --stride 64 --size 128

# train
python classification.py --mode train --data-dir ../data/train/processed/ --model-dir ../data/
python classification.py --mode train --data-dir ../data/train/processed/ --model-dir ../data/ --focal --alpha 0.9
python classification.py --mode train-synthesizer --data-dir ../data/train/processed/ --model-dir ../data/
python classification.py --mode train-synthesizer --data-dir ../data/train/processed/ --model-dir ../data/ --focal
python classification.py --mode train-multi --data-dir ../data/train/processed/ --model-dir ../data/
python classification.py --mode train-multi --data-dir ../data/train/processed/ --model-dir ../data/ --focal

# experiments
python classification.py --mode train --data-dir ../data/train/processed/ --model-dir ../data/ --experiments --focal
python classification.py --mode train-synthesizer --data-dir ../data/train/processed/ --model-dir ../data/ --experiments --focal
python classification.py --mode train-multi --data-dir ../data/train/processed/ --model-dir ../data/ --experiments --focal

# test
python classification.py --mode test --data-dir ../data/test/processed/ --model-dir ../data/
python classification.py --mode test-synthesizer --data-dir ../data/test/processed/ --model-dir ../data/
python classification.py --mode test-multi --data-dir ../data/test/processed/ --model-dir ../data/

# predict
python classification.py --mode predict --data-dir ../data/test/ --model-dir ../data/
python classification.py --mode predict-synthesizer --data-dir ../data/test/ --model-dir ../data/
python classification.py --mode predict-synthesizer --data-dir ../data/test/processed --model-dir ../data/ --device cuda
python classification.py --mode predict-multi --data-dir ../data/test/ --model-dir ../data/
python classification.py --mode predict-multi --data-dir ../data/test/processed --model-dir ../data/ --device cuda

# discriminate
python classification.py --mode test-synthesizer --data-dir ../data/test/processed/ --model-dir ../data/ --output-fake
python discriminator.py --mode train --data-dir ../data/test/processed --model-dir ../data
python classification.py --mode test-synthesizer --data-dir ../data/fine-tune/processed/ --model-dir ../data/ --output-fake

# fine-tune
python classification.py --mode train --data-dir ../data/fine-tune/processed/ --fine-tune --model-dir ../data/
python classification.py --mode train-synthesizer --data-dir ../data/fine-tune/processed/ --fine-tune --model-dir ../data/
python classification.py --mode test-synthesizer --data-dir ../data/fine-tune/processed/ --model-dir ../data/fine-tune/

# fine-tune-with-fake
python classification.py --mode train --data-dir ../data/fine-tune/processed/ --fine-tune-with-fake --model-dir ../data/ --batch-size 16
python classification.py --mode train-synthesizer --data-dir ../data/fine-tune/processed/ --fine-tune-with-fake --model-dir ../data/ --batch-size 16
python classification.py --mode test-synthesizer --data-dir ../data/fine-tune/processed/ --model-dir ../data/fine-tune

# unsupervised
python unsupervised.py --mode train --data-dir ../data/train/processed/ --model-dir ../data/
python unsupervised.py --mode predict --data-dir ../data/test/ --model-dir ../data/
python unsupervised.py --mode train --data-dir ../data/colored/train --model-dir ../data/ 
``` 
