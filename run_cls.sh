# @Author: Xuan Cao <xuan>
# @Date:   2019-10-28, 7:30:31
# @Last modified by:   xuan
# @Last modified time: 2019-11-13, 12:00:53



#train
# loss_weights: fc, dice, bce

# train cls
#python main_cls.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --step 2 --loss_weights '0,1,1' --model 'efficientnetb2-cls-BCE-HVSFR-W11-reduceLR' --fold 0

#python main_cls.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --loss_weights '0,2,1' --model 'resnet34-cls-BCE-HVSFR-W21-reduceLR' --fold 0

#python main_cls.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 32 --step 1 --loss_weights '0,3,1' --model 'resnet34-cls-BCE-HVSFR-W31-reduceLR' --fold 0

# predict cls
#python main_cls.py --mode 'predict_valid' --batch-size 32 --step 1 --loss_weights '0,1,1' --model 'resnet34-cls-BCE-HVSFR-W11-reduceLR' --fold 0

#python main_cls.py --mode 'predict_test' --batch-size 32 --step 1 --loss_weights '0,1,1' --model 'resnet34-cls-BCE-HVSFR-W11-reduceLR' --fold 0

# classifier

#python main_cls.py --n-epochs 15 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --pl 1 --step 1 --loss_weights '0,1,1' --model 'efficientnetb1-cls-BCE-HV-reduceLR-PL'


#python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb3-cls-BCE-HV-reduceLR' --fold 0

#python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb3-cls-BCE-HV-reduceLR' --fold 0

#python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb3-cls-BCE-HV-reduceLR' --fold 0 --tta 3

#python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb3-cls-BCE-HV-reduceLR' --fold 0 --tta 3

for i in {0..4}
do
  python main_cls.py --n-epochs 15 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --pl 1 --step 1 --loss_weights '0,1,1' --model 'efficientnetb0-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb0-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb0-cls-BCE-HV-reduceLR-PL' --fold $i --tta 3

  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb0-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb0-cls-BCE-HV-reduceLR-PL' --fold $i --tta 3

  python main_cls.py --n-epochs 30 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --pl 1 --step 1 --loss_weights '0,1,1' --model 'resnet50-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'resnet50-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'resnet50-cls-BCE-HV-reduceLR-PL' --fold $i --tta 3

  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'resnet50-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'resnet50-cls-BCE-HV-reduceLR-PL' --fold $i --tta 3

  python main_cls.py --n-epochs 15 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --pl 1 --step 1 --loss_weights '0,1,1' --model 'efficientnetb3-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb3-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb3-cls-BCE-HV-reduceLR-PL' --fold $i --tta 3

  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb3-cls-BCE-HV-reduceLR-PL' --fold $i

  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb3-cls-BCE-HV-reduceLR-PL' --fold $i --tta 3

done
