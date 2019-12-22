# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:44:08
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:11:51

# b5-Unet
for i in {0..4}
do
python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pl 0 --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16' --fold $i --tta 3
done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16' --tta 3

# inceptionResnetV2-FPN
for i in {0..4}
do
python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pl 0 --pos_only 1 --lrc 'warmRestart' --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --fold $i --tta 3
done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --tta 3

# b7-FPN
for i in {0..4}
do
python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pl 0 --pos_only 1 --lrc 'warmRestart' --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16' --fold $i --tta 3
done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16' --tta 3

# b7-Unet
for i in {0..4}
do
python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pl 0 --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'efficientnetb7-Unet-DICE-warmRestart-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'Unet' --model 'efficientnetb7-Unet-DICE-warmRestart-10x3-bs16' --fold $i --tta 3
done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet' --model 'efficientnetb7-Unet-DICE-warmRestart-10x3-bs16' --tta 3

# b7-FPN-PL
for i in {0..4}
do
python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pl 1 --pos_only 1 --lrc 'warmRestart' --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16-pl' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16-pl' --fold $i --tta 3
done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-10x3-bs16-pl' --tta 3
