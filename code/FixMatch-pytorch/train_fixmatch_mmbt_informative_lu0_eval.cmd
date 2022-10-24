SET DATA_PATH=D:\IUSTIN\DoinaData
python myTrain.py ^
--mu 1 ^
--k-img 9601 ^
--dropout 0.2 ^
--optimizer adam ^
--scheduler plateau ^
--my_format multimodal ^
--model mmbt ^
--train_file "train_eda" ^
--unlabeled_dataset "unlabeled_eda" ^
--unbalanced ^
--batch-size 1 ^
--gradient_accumulation_steps 16 ^
--epochs 50 ^
--dataset CrisisMMD ^
--task "informative_orig" ^
--lr 1e-5 ^
--threshold 0.7 ^
--use-ema ^
--text_soft_aug "none" ^
--text_hard_aug "eda_01" ^
--lambda-u 0 ^
--data_path "%DATA_PATH%\data_splits" ^
--img_path "%DATA_PATH%" ^
--resume "%DATA_PATH%\results\informative_g16_dr02_ema1_lu0_mu1_im+n+eda01_16may\model_best.pth.tar" ^
--eval_only
 