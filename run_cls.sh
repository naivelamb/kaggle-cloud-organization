# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 1:18:12
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:32:53



for i in {0..4}
do
  python main_cls.py --n-epochs 15 --patience 2 --lr 5e-4 --gamma 0.1 --lrc 'reduceLR' --batch-size 16 --pl 1 --step 1 --loss_weights '0,1,1' --model 'efficientnetb1-cls-BCE-reduceLR-bs16-PL' --fold $i

  python main_cls.py --mode 'predict_valid' --batch-size 16  --model 'efficientnetb1-cls-BCE-reduceLR-bs16-PL' --fold $i --tta 3

  python main_cls.py --mode 'predict_test' --batch-size 16 --model 'efficientnetb1-cls-BCE-reduceLR-bs16-PL' --fold $i --tta 3
done
