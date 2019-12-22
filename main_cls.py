# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:18:33
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:33:17



import os, argparse
import json
import shutil
import warnings
warnings.filterwarnings("ignore")
import time

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.exceptions import UndefinedMetricWarning

import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import segmentation_models_pytorch as smp

#from losses import FocalLoss, SoftDiceLoss

from dataSet.dataset import *
from dataSet.transforms import *
from dataSet.all_transforms import *
from models.cls_model import *
from util.utils import *
from util.lrs_schedulers import *
from common import *
from losses import *

from apex import amp
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=UserWarning)

def parse_arg():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', default='train')
    arg('--run_root', default='../output/')
    arg('--model', default='resnet34')
    arg('--clean', action='store_true')
    arg('--pretrained', type=int, default=0)
    arg('--batch-size', type=int, default=8)
    arg('--step', type=int, default=4)
    arg('--gamma', type=float, default=0.5)
    arg('--gamma-step', type=int, default=3)
    arg('--patience', type=int, default=3)
    arg('--lr', type=float, default=3e-4)
    arg('--lrc', type=str, default='reduceLR')
    arg('--workers', type=int, default=2 if ON_KAGGLE else 8)
    arg('--n-epochs', type=int, default=50)
    arg('--tta', type=int, default=1)
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)
    arg('--pl', type=int, default=0)
    arg('--loss_weights', type=str, default='0,1,1')
    arg('--sliding', type=int, default=0)
    arg('--sampling', type=int, default=0)
    arg('--framework', type=str, default='Unet')
    arg('--channel_wise', type=int, default=0)
    arg('--annealing_epoch', type=int, default=5)
    arg('--pixel', type=str, default='0.5,0.5,0.5,0.5')
    arg('--area', type=str, default='0,0,0,0')

    args = parser.parse_args()
    args.loss_weights = [float(x) for x in args.loss_weights.split(',')]
    args.pixel = [float(x) for x in args.pixel.split(',')]
    args.area = [float(x) for x in args.area.split(',')]
    return args

def main():
    args = parse_arg()
    set_seed(1217)
    print(args.model)
    print('%s fold-%d...' % (args.mode, args.fold))
    args.run_root = args.run_root  + '/' + args.model#'0822_efficientnetb0_LB804'
    run_root = Path(args.run_root)

    if run_root.exists() and args.clean:
        shutil.rmtree(run_root)
    run_root.mkdir(exist_ok=True, parents=True)

    train_root = DATA_ROOT / ('images_%d' % SIZE[0])
    valid_root = train_root
    test_root = train_root

    sample_sub = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
    ss = pd.DataFrame()
    ss['Image_Label'] = sample_sub['Image_Label'].apply(lambda x: x.split('_')[0]).unique()
    ss['EncodedPixels'] = '1 1'

    fold_df = pd.read_csv('./files/5-folds_%d.csv' % (SIZE[0]))
    train_fold = fold_df[fold_df['fold']!=args.fold].reset_index(drop=True)
    valid_fold = fold_df[fold_df['fold']==args.fold].reset_index(drop=True)

    PIXEL_THRESHOLDS = args.pixel
    AREA_SIZES = args.area

    if args.pl == 1: # add puesdo label
        df_pl = pd.read_csv('./files/df_pl.csv')
        for col in train_fold.columns:
            if col not in df_pl.columns:
                df_pl[col] = 0
        train_fold = train_fold.append(df_pl)
        train_fold.fillna('', inplace=True)

    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    if args.sliding:
        train_transform = transform_train_al((256, 256))
    else:
        train_transform = transform_train_al(SIZE)
    test_transform = transform_test_al(SIZE)

    model_name = args.model if '-' not in args.model else args.model.split('-')[0]
    if model_name.startswith('effi'):
        model_name = model_name[:-2] + '-' + model_name[-2:]
        model = efficientnet(model_name, num_classes=NUM_CLASSES)
    elif model_name.startswith('resnet'):
        model = resnet(model_name, num_classes=NUM_CLASSES)
    if args.mode == 'train':
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        training_set = Dataset_cloud(train_root,
                                     df=train_fold, transform=train_transform,
                                     mode='train')
        #sampler = EmptySampler(data_source=training_set, positive_ratio_range=sampler_ratio, epochs=args.n_epochs)
        validation_set = Dataset_cloud(train_root,
                                       df=valid_fold, transform=test_transform,
                                       mode='train')

        print(f'{len(training_set):,} items in train, ', f'{len(validation_set):,} in valid')


        train_loader = DataLoader(training_set,
                                  batch_size=args.batch_size,
                                  num_workers=args.workers,
                                  sampler=None,
                                  drop_last=False,
                                  shuffle=True,)
        valid_loader = DataLoader(validation_set, shuffle=False,
                                  batch_size=args.batch_size,
                                  #collate_fn=null_collate,
                                  num_workers=args.workers)

        model = model.cuda()

        #optimizer = Adam([{'params': model.encoder.parameters(), 'lr': args.lr},
        #                  {'params': model.decoder.parameters(), 'lr': args.lr*10}])
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=args.lr, weight_decay=0, betas=(0.9, 0.999), eps=1e-08)
        if args.lrc == 'reduceLR':
            scheduler = ReduceLROnPlateau(optimizer, patience=args.patience,
                                          factor=args.gamma, verbose=True, mode='max')
        elif args.lrc == 'cos':
            scheduler = CosineAnnealingLR(optimizer, args.patience, eta_min=args.lr*args.gamma)
        elif args.lrc == 'warmRestart':
            scheduler = WarmRestart(optimizer, T_max=args.patience, T_mult=1, eta_min=1e-6)
#        scheduler = StepLR(optimizer, step_size=args.patience, gamma=args.gamma)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
#
        train_kwargs = dict(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            # use_cuda=use_cuda,
            epoch_length=len(training_set),
        )

        train(n_epochs=args.n_epochs, **train_kwargs)

        file = '%s/train-%d.log'%(args.run_root, args.fold)
        df = pd.read_csv(file, sep='|')
        cols = df.columns
        df.columns = [x.strip() for x in cols]
        fig, ax = plt.subplots(2, 2, figsize=(12,12))
        #loss profile
        ax[0, 0].plot(df.epoch, df.loss, label='train-loss', marker='o')
        ax[0, 0].plot(df.epoch, df['val loss'], label='val-loss', marker='x')
        ax[0, 0].set_xlabel('epoch')
        ax[0, 0].set_ylabel('loss')
        ax[0, 0].legend()
        #lr profile
        ax[0, 1].plot(df.epoch, df.lr, label='lr', marker='o')
        ax[0, 1].set_xlabel('epoch')
        ax[0, 1].set_ylabel('lr')
        ax[0, 1].legend()
        if 'AUC-mean' in df.columns: #cls
            ax[1, 0].plot(df.epoch, df['AUC-mean'], '-ro', label='AUC-mean')
            ax[1, 0].set_xlabel('epoch')
            ax[1, 0].set_ylabel('AUC-mean')
            ax[1, 0].legend()
            for k in range(4):
                ax[1, 1].plot(df.epoch, df['class%d'%(k+1)], '-o', label=CLASS_NAMES[k])
            ax[1, 1].set_xlabel('epoch')
            ax[1, 1].set_ylabel('AUC')
            ax[1, 1].legend()
        else:
            ax[1, 0].plot(df.epoch, df['val dice'], '-ro', label='dice')
            ax[1, 0].set_xlabel('epoch')
            ax[1, 0].set_ylabel('val-dice')
            ax[1, 0].legend()
        fig.savefig(Path(args.run_root)/('train_%d.png'%(args.fold)))

    elif args.mode.startswith('predict'):
        if (run_root /('best-dice-%d.pt' % args.fold)).exists():
            load_model(model, run_root /('best-dice-%d.pt' % args.fold), multi2single=False)
        else:
            load_model(model, run_root /('best-model-%d.pt' % args.fold), multi2single=False)
        model = model.cuda()
        if args.mode == 'predict_valid':
            valid_set = Dataset_cloud(valid_root,
                                      df=valid_fold, transform=test_transform,
                                      mode='test')
            valid_loader = DataLoader(valid_set, shuffle=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.workers)
            predict(model, args.mode, loader=valid_loader, out_path=run_root, fold=args.fold, tta=args.tta)

        elif args.mode == 'predict_test':
            if args.limit:
                ss = ss[:args.limit]
            test_set = Dataset_cloud(test_root,
                                     df=ss, transform=test_transform,
                                     mode='test')
            test_loader = DataLoader(test_set, shuffle=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.workers)
            predict(model, args.mode, loader=test_loader, out_path=run_root, fold=args.fold, tta=args.tta)
        elif args.mode == 'predict_5fold':
            if args.limit:
                ss = ss[:args.limit]
            test_set = Dataset_cloud(test_root,
                                     df=ss, transform=test_transform,
                                     mode='test')
            test_loader = DataLoader(test_set, shuffle=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.workers)
            predict_5fold(test_loader, out_path=run_root, args=args,
                          pixel_thresholds=PIXEL_THRESHOLDS,
                          area_size=AREA_SIZES)
        else:
            RuntimeError('%s mode not implemented' % (args.mode))
    elif args.mode == 'opt':
        # creat save folder
        if (run_root /('opt')).exists():
            pass
        else:
            output_root = Path(run_root /('opt'))
            output_root.mkdir(exist_ok=True, parents=True)
        # Load model
        if (run_root /('best-dice-%d.pt' % args.fold)).exists():
            load_model(model, run_root /('best-dice-%d.pt' % args.fold), multi2single=False)
        else:
            load_model(model, run_root /('best-model-%d.pt' % args.fold), multi2single=False)
        model = model.cuda()

        valid_set = Dataset_cloud(valid_root,
                                  df=valid_fold, transform=test_transform,
                                  mode='test')
        valid_loader = DataLoader(valid_set, shuffle=False,
                                 batch_size=args.batch_size,
                                 num_workers=args.workers)

        area_ts_list = [0,0,0,0]
        for pixel_ts in range(0, 80, 5):
            pixel_ts /= 100
            pixel_ts_list = [pixel_ts] * 4
            print('Processing: pixel-[%s]'%(str(pixel_ts)))
            predict(model, args.mode, loader=valid_loader,
                    out_path=run_root /('opt'),
                    fold=args.fold, tta=args.tta,
                    pixel_thresholds=pixel_ts_list,
                    area_size=area_ts_list,
                    )
    else:
        print('%s mode not implemented' % (args.mode))

def train(args, model: nn.Module, optimizer, scheduler, *,
          train_loader, valid_loader, epoch_length,  # use_cuda,
          n_epochs=None, strategy=None) -> bool:
    n_epochs = n_epochs or args.n_epochs

    run_root = Path(args.run_root)
    model_path = run_root / ('model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    best_cls_path = run_root / ('best-dice-%d.pt' % args.fold)

    if best_cls_path.exists():
        state, best_valid_loss = load_model(model, best_cls_path)
        val_auc = state['best_valid_auc']
        best_valid_loss = state['best_valid_loss']
        start_epoch = state['epoch']
        step = state['step']
        best_epoch = state['epoch']
    else:
        best_valid_auc = 0
        best_valid_loss = 999999
        start_epoch = 0
        step = 0
        best_epoch = 0
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_valid_auc': val_auc,
        'best_valid_loss': val_loss,
    }, str(model_path))

    report_each = 1000000
    if (run_root / ('train_%d.log' % (args.fold))).exists():
        log = run_root.joinpath('train-%d.log' % args.fold).open('at', encoding='utf8')
    else:
        log = run_root.joinpath('train-%d.log' % args.fold).open('at', encoding='utf8')
        log.write('epoch|   lr   |  loss |val loss|AUC-mean|class1|class2|class3|class4|neg_F1|time|save\n')

    n_circle = 0
    if isinstance(scheduler, WarmRestart):
        n_annealing = args.annealing_epoch
    else:
        n_annealing = 0
    for epoch in range(start_epoch, start_epoch + n_epochs + n_annealing):
        start_time = time.time()
        model.train()
        lr = get_learning_rate(optimizer)
        tq = tqdm.tqdm(total=epoch_length, ascii=True)
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []
        mean_loss = 0
        if args.sampling:
            train_loader.sampler.set_epoch(epoch)
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size,channel,_,_ = targets.shape
            targets_fc = (targets.view(batch_size, channel, -1).sum(-1)>0).float()
            #optimizer.zero_grad()
            #with torch.set_grad_enabled(True):
            logits = model(inputs)
            logits_fc = logits.view(batch_size, channel, -1)
            logits_fc = torch.max(logits_fc, -1)[0]

            loss = WeightedBCE(weights=args.loss_weights[-2:])(logits_fc, targets_fc)
            #loss.backward()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            if (i+1) % args.step == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                tq.update(batch_size*args.step)

                losses.append(loss.item())

                #running_loss += loss.item() * inputs.size(0)
                mean_loss = np.mean(losses[-report_each:])
                #tq.set_postfix(loss=(running_loss / ((i+1) * batch_size)))
                tq.set_postfix(loss=f'{mean_loss:.5f}')


            if i and i % report_each == 0:
                write_event(log, step, loss=mean_loss)

        ############## On Epoch End ####################
        # valid -> save -> save best
        #write_event(log, step, epoch=epoch, loss=mean_loss)
        tq.close()
        valid_metrics = validation(model, valid_loader, args, save_result=True)
        #write_event(log, step, epoch, **valid_metrics)
        val_loss = valid_metrics['val_loss']
        aucs = valid_metrics['aucs']
        val_f1 = valid_metrics['F1']
        val_auc = np.mean(valid_metrics['aucs'])
        _save_ckp = ' '

        save(epoch + 1)

        if epoch < start_epoch + n_epochs - 1:
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif isinstance(scheduler, StepLR):
                    scheduler.step()
                elif isinstance(scheduler, GradualWarmupScheduler):
                    scheduler.step()
                elif isinstance(scheduler, CosineAnnealingLR):
                    scheduler.step()
                elif isinstance(scheduler, WarmRestart):
                    if epoch != 0:
                        scheduler.step()
                        scheduler=warm_restart(scheduler, T_mult=2)
                else:
                    raise RuntimeError('Opeartion for scheduler not implemented.')
        elif epoch < start_epoch + n_epochs + 2 and epoch >= start_epoch + n_epochs - 1:
            optimizer.param_groups[0]['lr'] = 1e-5
        else:
            optimizer.param_groups[0]['lr'] = 5e-6

        if val_auc > best_valid_auc:
            best_valid_auc = val_auc
            shutil.copy(str(model_path), str(best_cls_path))
            _save_ckp += '#'
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            #best_valid_dice = val_dice
            shutil.copy(str(model_path), str(best_model_path))
            best_epoch = epoch
            _save_ckp += '*'
        run_time = (time.time() - start_time)/60
        #epoch|   lr   |  loss |val loss|class1|class2|class3|class4|neg_F1|time|save
        log.write('%5d|%1.6f|%.5f| %.5f| %.5f|%.4f|%.4f|%.4f|%.4f|%.4f|%.2f|%s' \
                  % (epoch, lr, mean_loss, val_loss, val_auc, aucs[0], aucs[1], aucs[2],
                     aucs[3], valid_metrics['F1'], run_time, _save_ckp))
        run_time = (time.time() - start_time)/60

        log.write('\n')
        log.flush()
    print('Best epoch: %d, Loss: %.5f, AUC: %.5f'\
          % (best_epoch, best_valid_loss, best_valid_auc))
    return True

def validation(model: nn.Module, valid_loader, args, save_result=False) -> Dict[str, float]:
    run_root = Path(args.run_root)
    model.eval()
    all_losses = 0
    cls_targets, cls_pred = [], []
    n_samples = 0
    with torch.no_grad():
        for i, (inputs, targets) in tqdm.tqdm(enumerate(valid_loader)):
            batch_size, channel, _, _ = targets.size()
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_fc = (targets.view(batch_size, channel, -1).sum(-1) > 0).float()
            cls_targets.append(targets_fc.cpu().numpy())

            if args.sliding:
                logits, logits_fc = predict_sliding(model, inputs)
            else:
                logits_fc = model(inputs)
            #print(logits_fc.shape)
            cls_pred.append((F.sigmoid(logits_fc)).cpu().numpy())
            loss = WeightedBCE(weights=args.loss_weights[-2:])(logits_fc, targets_fc)

            all_losses += loss.item()*batch_size
            n_samples += batch_size

    cls_targets = np.concatenate(cls_targets)
    cls_pred = np.concatenate(cls_pred)


    metrics = {}
    metrics['val_loss'] = float(all_losses/n_samples)

    metrics['hit'], metrics['aucs'], [neg_precision, neg_recall] = metric_hit(cls_pred, cls_targets)
    metrics['F1'] = 2/(1/neg_precision + 1/neg_recall)
    to_print = []
    for idx, (k, v) in enumerate(metrics.items()):
        if k == 'dice_detail':
            v_p = ['%.3f' % (x) for x in v]
            v_p = ','.join(v_p)
            to_print.append(f'{k} [{v_p}]')
        elif k == 'aucs':
            to_print.append('Mean auc %.5f' % (np.mean(metrics['aucs'])))
            v_p = ['%.3f' % (x) for x in v]
            v_p = ','.join(v_p)
            to_print.append(f'{k} [{v_p}]')
        elif k == 'hit':
            pass
        else:
            to_print.append(f'{k} {v:.3f}')
    #to_print.append(str(np.sum(pred)))
    print(' | '.join(to_print))
    return metrics

def predict(model, mode, loader, out_path: Path, fold, tta):
    mode = mode.split('_')[-1]
    model.eval()
    all_outputs, all_ids, all_cls = [], [], []
    with torch.no_grad():
        for inputs, names in tqdm.tqdm(loader, desc='Predict'):
            inputs = inputs.cuda()
            outputs = model(inputs)
            cls_pred = F.sigmoid(outputs).view(-1).data.cpu().numpy()

            if tta >= 2:# h flip
                outputs = model(inputs.flip(3))
                cls_pred += F.sigmoid(outputs).view(-1).data.cpu().numpy()
            if tta == 3:# v flip
                outputs = model(inputs.flip(2))
                cls_pred += F.sigmoid(outputs).view(-1).data.cpu().numpy()
            if tta != 0:
                cls_pred /= tta
            all_cls.append(cls_pred)

            ids = [item for sublist in list(zip(*names)) for item in sublist]
            all_ids.extend(ids)

    df = pd.DataFrame(data=np.concatenate(all_cls), index=all_ids)
    df = mean_df(df).reset_index()
    df.rename(columns={'index': 'Image_Label'}, inplace=True)
    df['EncodedPixels'] = np.nan
    df['EncodedPixels'].loc[df[0] > 0.5] = '1 1'

    if tta <= 1:
        df.to_csv(out_path / ('%s_cls_fold%d.csv' % (mode, fold)), index=None)
    else:
        df.to_csv(out_path / ('%s_cls_fold%d_tta%d.csv' % (mode, fold, tta)), index=None)

    print(f'Saved predictions to {out_path}')


if __name__ == '__main__':
    main()
