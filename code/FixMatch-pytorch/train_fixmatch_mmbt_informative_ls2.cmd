SET DATA_PATH=D:\IUSTIN\DoinaData
python myTrain.py ^
--mu 7 ^
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
--text_hard_aug "none" ^
--linear_lu ^
--distil unlabeled ^
--lambda-u 100 ^
--data_path "%DATA_PATH%\data_splits" ^
--img_path "%DATA_PATH%" ^
--resume "%DATA_PATH%\results\informative_g16_dr02_ema1_lu+2_mu7_im+n_17may\model_best.pth.tar" ^
--eval_only
