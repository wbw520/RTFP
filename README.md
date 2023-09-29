# Facade Parsing Via Vision Transformer and Line Revision

## Model Structure
![Structure Figure](figs/Figure_overview.png)

## Data Preparation
Our data is avaliable by contacting wang@ids.osaka-u.ac.jp
Using following command to complie our dataset
```
cd data/
python facade_data_generation.py --root [your_root]
```

## Training
```
python main.py --model_name Segmenter --batch_size 4 --root [your_root]
```

## Inference
```
python inference.py --model_name Segmenter --eval_img[your_root]
```
