python simple_extractor.py --dataset atr --model-restore ./pretrain_model/atr.pth  --input-dir ./images --output-dir ./output/atr

python simple_extractor.py --dataset lip --model-restore ./pretrain_model/lip.pth  --input-dir ./images --output-dir ./output/lip

python simple_extractor.py --dataset pascal --model-restore ./pretrain_model/pascal.pth  --input-dir ./images --output-dir ./output/pascal
