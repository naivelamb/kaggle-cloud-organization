# @Author: Xuan Cao <x0c02jg>
# @Date:   2019-10-25, 9:31:30
# @Email:  xuan.cao@walmartlabs.com
# @Last modified by:   x0c02jg
# @Last modified time: 2019-11-15, 1:09:25



#train
# loss_weights: fc, dice, bce

#for i in {0..4}
#do
#  python main_seg.py --n-epochs 30 --patience 2 --lr 1e-3 --gamma 0.1 --batch-size 32 --step 1 --loss_weights '0,0,1' --lrc 'reduceLR' --framework 'Unet' --model 'resnet34-Unet-BCE-HV-reduceLR(1e-3)-0' --fold $i

#  python main_seg.py --mode 'opt' --batch-size 32 --framework 'Unet' --model 'resnet34-Unet-BCE-HV-reduceLR(1e-3)-0' --fold $i

#  python main_seg.py --mode 'opt' --batch-size 32 --tta 3 --framework 'Unet' --model 'resnet34-Unet-BCE-HV-reduceLR(1e-3)-0' --fold $i
#done

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'efficientnetb0-Unet-DICE-warmRestart-bs16-t0'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'efficientnetb3-Unet-DICE-warmRestart-bs16-t0'
#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-bs16-t0'

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-bs16-t0' --tta 3

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'se_resnext101_32x4d-Unet-DICE-warmRestart-10x3-bs16'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,0,1' --lrc 'warmRestart' --framework 'FPN' --model 'resnet50-FPN-BCE-warmRestart-10x3-bs16'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,0,1' --lrc 'warmRestart' --framework 'FPN' --model 'efficientnetb1-FPN-BCE-warmRestart-10x3-bs16'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0.1,0,3' --lrc 'warmRestart' --framework 'Unet_ds' --model 'resnet101-Unet_ds-BCE-warmRestart-10x3-bs16'

#python main_seg.py --n-epochs 40 --patience 2 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'reduceLR' --framework 'Unet' --model 'se_resnext101_32x4d-Unet-DICE-reduceLR-bs16' --fold 0

#python main_seg.py --n-epochs 20 --patience 2 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'reduceLR' --framework 'FPN' --model 'efficientnetb7-FPN-DICE-reduceLR-bs16-t1'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'inceptionresnetv2-Unet-BCE-warmRestart-10x3-bs16'

#python main_seg.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'reduceLR' --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-reduceLR-10x3-bs16-t0'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'FPN' --model 'se_resnext101_32x4d-FPN-DICE-warmRestart-10x3-bs16'

#python main_seg.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'reduceLR' --framework 'FPN' --model 'se_resnext101_32x4d-FPN-DICE-reduceLR-10x3-bs16'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --fold 1

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --fold 1

#python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --fold 0

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --fold 1 --tta 3

#python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'FPN' --model 'inceptionresnetv2-FPN-DICE-warmRestart-10x3-bs16' --fold 0 --tta 3

python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16'

python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16' --tta 3

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16'

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16' --tta 3

for i in {1..4}
do
python main_seg.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,0,1' --pl 0 --pos_only 0 --lrc 'warmRestart' --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16' --fold $i --tta 3

done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16'

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet' --model 'densenet121-Unet-BCE-reduceLR-10x3-bs16' --tta 3


python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16'

python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16' --tta 3

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16'

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16' --tta 3

for i in {1..4}
do
python main_seg.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0.1,0,1' --pl 0 --pos_only 0 --lrc 'warmRestart' --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16' --fold $i --tta 3

done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16'

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16' --tta 3

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16'

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet_ds' --model 'densenet121-Unet_ds-BCE-reduceLR-10x3-bs16' --tta 3
#for i in {0..4}
#do
#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pl 1 --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16-pl' --fold $i

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16-pl' --fold $i

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16-pl' --fold $i --tta 3

#done

#python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16-pl'

#python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'Unet' --model 'efficientnetb5-Unet-DICE-warmRestart-10x3-bs16-pl' --tta 3
#python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'FPN' --model 'densenet201-FPN-BCE-warmRestart-bs16'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'densenet169-Unet-DICE-warmRestart-bs16'

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'Unet' --model 'densenet169-Unet-DICE-warmRestart-bs16'

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'FPN' --model 'densenet161-FPN-BCE-warmRestart-bs16' --tta 3

#python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'Unet' --model 'densenet169-Unet-DICE-warmRestart-bs16'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-bs16'

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-bs16'

#python main_seg.py --mode 'predict_valid' --batch-size 16  --framework 'FPN' --model 'densenet169-FPN-BCE-warmRestart-bs16' --tta 3

#python main_seg.py --mode 'predict_test' --batch-size 16  --framework 'FPN' --model 'efficientnetb7-FPN-DICE-warmRestart-bs16'
#python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'FPN' --model 'densenet169-FPN-BCE-warmRestart-bs16' --tta 3

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,1,0' --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'efficientnetb0-Unet-DICE-warmRestart-bs16-t0'
#python main_seg.py --mode 'predict_test' --batch-size 32 --framework 'FPN' --model 'resnet34-FPN-BCE-HV-warmRestart(5x3)'
#python main_cls.py --n-epochs 20 --patience 1 --lr 1e-3 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --loss_weights '0,1,1' --model 'resnet34-cls-BCE-HV-reduceLR'

#python main_cls.py --mode 'predict_valid' --batch-size 32  --model 'resnet34-cls-BCE-HVSFR-W11-reduceLR' --fold 0 --tta 3

#python main_cls.py --mode 'predict_test' --batch-size 32 --model 'resnet34-cls-BCE-HVSFR-W11-reduceLR' --fold 0 --tta 3

#python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold 0

#python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold 0 --tta 3

#python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold 0

#python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold 0 --tta 3

#for i in {1..4}
#do
#  python main_cls.py --n-epochs 15 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --step 2 --loss_weights '0,1,1' --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold $i

#  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold $i

#  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold $i --tta 3

#  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold $i

#  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb1-cls-BCE-HVSFR-W11-reduceLR-t0' --fold $i --tta 3
#done

#python main_cls.py --n-epochs 10 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --step 2 --loss_weights '0,1,1' --model 'efficientnetb0-cls-BCE-HVSFR-W11-reduceLR-t0' --fold 0
