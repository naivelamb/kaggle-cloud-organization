# @Author: Xuan Cao <xuan>
# @Date:   2019-10-28, 8:01:26
# @Last modified by:   xuan
# @Last modified time: 2019-11-15, 2:17:55



#train
# loss_weights: fc, dice, bce

#python main_seg.py --n-epochs 15 --patience 1 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 0 --pos_only 1 --loss_weights '0,2,1' --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-W21-c0'

#python main_seg.py --n-epochs 30 --patience 30 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,0,1' --lrc 'warmRestart' --framework 'FPN' --model 'se_resnext50_32x4d-FPN-BCE-warmRestart-bs16-t2'
#python main_seg.py --n-epochs 15 --patience 30 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0.1,0,1' --lrc 'warmRestart' --framework 'FPN_ds' --model 'resnet34-FPN_ds-BCE-warmRestart-bs16-t0'

#python main_seg.py --n-epochs 15 --patience 30 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0.2,0,1' --lrc 'warmRestart' --framework 'FPN_ds' --model 'resnet34-FPN_ds-BCE-warmRestart-bs16-t1'

## main_seg.py --n-epochs 15 --patience 30 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0.5,0,1' --lrc 'warmRestart' --framework 'FPN_ds' --model 'resnet34-FPN_ds-BCE-warmRestart-bs16-t2'

#python main_seg.py --mode 'predict_valid' --batch-size 16 --step 1 --framework 'Unet' --model 'efficientnetb0-Unet-BCE-warmRestart-bs16-t0'

#python main_seg.py --mode 'predict_test' --batch-size 16 --step 1 --framework 'Unet' --model 'efficientnetb0-Unet-BCE-warmRestart-bs16-t0'


#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '1,0,1' --lrc 'warmRestart' --framework 'Unet_ds' --model 'efficientnetb0-Unet_ds-BCE-warmRestart-bs16-t0'

#python main_seg.py --n-epochs 60 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0.1,1,3' --lrc 'warmRestart' --framework 'Unet_ds' --model 'resnet34-Unet_ds-BCE-warmRestart-bs16-t8'

#python main_seg.py --n-epochs 60 --patience 5 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0.1,1,3' --lrc 'warmRestart' --framework 'Unet_ds' --model 'resnet34-Unet_ds-BCE-warmRestart-bs16-t9'

#python main_seg.py --n-epochs 60 --patience 5 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,3,1' --pos_only 1 --lrc 'warmRestart' --framework 'Unet' --model 'resnet34-Unet-BCEDICE-warmRestart-bs16-t0'

#python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,0,1' --lrc 'warmRestart' --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16'

# python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16'
#
# python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16' --tta 3
#
# python main_seg.py --mode 'predict_test' --batch-size 16 --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16'
#
# for i in {1..4}
# do
#
# python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,0,1' --lrc 'warmRestart' --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16' --fold $i
#
# python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16' --fold $i
#
# python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16' --tta 3 --fold $i
#
# #python main_seg.py --mode 'predict_test' --batch-size 16 --framework 'UNet' --model 'efficientnetb7-Unet-DICE-warmRestart-10x3-bs16' --fold $i
#
# #python main_seg.py --mode 'predict_test' --batch-size 16 --framework 'UNet' --model 'efficientnetb7-Unet-DICE-warmRestart-10x3-bs16' --tta 3 --fold $i
# #python main_seg.py --mode 'predict_test' --batch-size 16 --framework 'FPN' --model 'resnet34-FPN-BCE-warmRestart-bs16-t1'
# done

python main_seg.py --mode 'predict_5fold' --batch-size 16 --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16'

python main_seg.py --mode 'predict_5fold' --batch-size 16 --framework 'JPU' --model 'resnet34-JPU-BCE-warmRestart-bs16' --tta 3

#python 2-stage-ensemble.py
#python main_seg.py --n-epochs 15 --patience 30 --lr 5e-4 --gamma 0.1 --lrc 'warmRestart' --batch-size 8 --step 1 --pos_only 0 --loss_weights '0,1,1' --framework 'FPN' --model 'efficientnetb5-FPN-BCEDICE-HVSSRGD-warmRestart-W011' --fold 0

#python main_seg.py --n-epochs 15 --patience 30 --lr 5e-4 --gamma 0.1 --lrc 'warmRestart' --batch-size 8 --step 2 --pos_only 1 --loss_weights '0,3,1' --framework 'FPN' --model 'efficientnetb5-FPN-BCEDICE-HVSSRGD-warmRestart-W31-t1' --fold 0

#python main_seg.py --n-epochs 15 --patience 30 --lr 5e-4 --gamma 0.1 --lrc 'warmRestart' --batch-size 8 --step 2 --pos_only 0 --loss_weights '0,1,1' --framework 'FPN' --model 'efficientnetb5-FPN-BCEDICE-HVSSRGD-warmRestart-W011-step2' --fold 0

#python main_seg.py --n-epochs 15 --patience 20 --lr 5e-4 --gamma 0.1 --lrc 'warmRestart' --batch-size 8 --step 2 --pos_only 0 --loss_weights '0,0,1' --framework 'FPN' --model 'efficientnetb5-FPN-BCE-HVSSRGD-warmRestart-W001' --fold 1

#python main_seg.py --n-epochs 15 --patience 20 --lr 5e-4 --gamma 0.1 --lrc 'warmRestart' --batch-size 8 --step 2 --pos_only 0 --loss_weights '0,0,1' --framework 'FPN' --model 'efficientnetb5-FPN-BCE-HVSSRGD-warmRestart-W001' --fold 2

#python main_seg.py --n-epochs 15 --patience 20 --lr 5e-4 --gamma 0.1 --lrc 'warmRestart' --batch-size 8 --step 2 --pos_only 0 --loss_weights '0,0,1' --framework 'FPN' --model 'efficientnetb5-FPN-BCE-HVSSRGD-warmRestart-W001' --fold 3

#python main_seg.py --n-epochs 15 --patience 20 --lr 5e-4 --gamma 0.1 --lrc 'warmRestart' --batch-size 8 --step 2 --pos_only 0 --loss_weights '0,0,1' --framework 'FPN' --model 'efficientnetb5-FPN-BCE-HVSSRGD-warmRestart-W001' --fold 4
#python main_seg.py --n-epochs 20 --patience 1 --lr 1e-3 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --channel_wise 1 --loss_weights '0,0,1' --framework 'FPN' --model 'resnet34-FPN-BCE-HV-reduceLR-20x1'

#python main_seg.py --n-epochs 40 --patience 1 --lr 1e-3 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --channel_wise 1 --loss_weights '0,1,1' --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-40x1'

#python main_seg.py --n-epochs 40 --patience 20 --lr 1e-3 --gamma 0.1 --lrc 'warmRestart' --batch-size 32 --step 1 --channel_wise 1 --loss_weights '0,1,1' --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-warmRestart-20x2'

#python main_seg.py --mode 'predict_test' --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 0 --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c0'

#python main_seg.py --n-epochs 15 --patience 1 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 1 --pos_only 1 --loss_weights '0,1,1' --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c1'

#python main_seg.py --mode 'predict_valid' --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 1 --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c1'

#python main_seg.py --mode 'predict_test' --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 1 --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c1'

#python main_seg.py --n-epochs 15 --patience 1 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 2 --pos_only 1 --loss_weights '0,1,1' --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c2'

#python main_seg.py --mode 'predict_valid' --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 2 --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c2'

#python main_seg.py --mode 'predict_test' --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 2 --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c2'

#python main_seg.py --n-epochs 15 --patience 1 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 3 --pos_only 1 --loss_weights '0,1,1' --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c3'

#python main_seg.py --mode 'predict_valid' --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 3 --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c3'

#python main_seg.py --mode 'predict_test' --lrc 'reduceLR' --batch-size 32 --step 1 --cls_label 3 --framework 'FPN' --model 'resnet34-FPN-BCEDICE-HV-reduceLR-c3'


#
#for i in {0..4}
#do
#  python main_seg.py --n-epochs 20 --patience 1 --lr 1e-3 --gamma 0.1 --batch-size 32 --step 1 --loss_weights '0,0,1' --lrc 'reduceLR' --framework 'FPN' --model 'resnet34-FPN-BCE-HV-reduceLR-20x1' --fold $i

  #python main_seg.py --mode 'opt' --batch-size 32 --framework 'FPN' --model 'resnet34-FPN-BCE-HV-reduceLR(1e-3)-0' --fold $i

  #python main_seg.py --mode 'opt' --batch-size 32 --tta 3 --framework 'FPN' --model 'resnet34-FPN-BCE-HV-reduceLR(1e-3)-0' --fold $i
#done

#python main_seg.py --mode 'predict_5fold' --batch-size 32 --step 1 --tta 3 --framework 'FPN' --model 'resnet34-FPN-BCE-HV-reduceLR(1e-3)-0' --pixel '0.35,0.4,0.35,0.35' --area '28000,20000,26000,14000'

#predict test
#python main_seg.py --mode 'predict_test' --batch-size 32 --framework 'FPN' --model 'resnet34-FPN-BCE-HV-reduceLR-20x1' --fold 4

#predict valid
#python main_seg.py --mode 'predict_valid' --batch-size 32  --framework 'FPN' --model 'resnet34-FPN-BCE-HV-reduceLR-15x1-addNormalization'
