# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:44:08
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:13:26


for i in {0..4}
do
python main_seg.py --n-epochs 30 --patience 10 --lr 5e-4 --gamma 0.1 --batch-size 16 --step 1 --loss_weights '0,0,1' --pl 0 --pos_only 0 --lrc 'warmRestart' --framework 'FPN' --model 'densenet121-FPN-BCE-warmRestart-10x3-bs16' --fold $i

python main_seg.py --mode 'predict_valid' --batch-size 16 --framework 'FPN' --model 'densenet121-FPN-BCE-warmRestart-10x3-bs16' --fold $i
done

python main_seg.py --mode 'predict_5fold' --batch-size 16  --framework 'FPN' --model 'densenet121-FPN-BCE-warmRestart-10x3-bs16'
