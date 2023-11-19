# Improving Facade Parsing with Vision Transformers and Line Integration
[PDF](https://arxiv.org/pdf/2309.15523.pdf)

## Model Structure
![Structure Figure](figs/Figure_overview.png)

## Data Preparation
[Download](https://drive.google.com/file/d/1KWwRhjwuJHBh_sez2tvKlWjFXjXAe_CZ/view?usp=drive_link)
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
python inference.py --model_name Segmenter --eval_img [your_root]
```
