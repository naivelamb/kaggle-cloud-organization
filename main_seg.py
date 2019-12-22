# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:18:33
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:32:30



import os, argparse
import json
import shutil
import warnings
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
from models.model import *
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
    arg('--loss_weights', type=str, default='0,0,1')
    arg('--sliding', type=int, default=0)
    arg('--sampling', type=int, default=0)
    arg('--framework', type=str, default='Unet')
    arg('--channel_wise', type=int, default=0)
    arg('--annealing_epoch', type=int, default=5)
    arg('--pixel', type=str, default='0.5,0.5,0.5,0.5')
    arg('--area', type=str, default='0,0,0,0')
    arg('--cls_label', type=int, default=4)
    arg('--pos_only', type=int, default=0)
    arg('--loss_per_image', type=int, default=0)

    args = parser.parse_args()
    args.loss_weights = [float(x) for x in args.loss_weights.split(',')]
    args.pixel = [float(x) for x in args.pixel.split(',')]
    args.area = [float(x) for x in args.area.split(',')]
    if args.cls_label > 3:
        args.cls_label = None
        args.n_classes = NUM_CLASSES
    else:
        args.n_classes = 1
        args.channel_wise=0
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

    if args.pos_only and args.cls_label in [0,1,2,3]:
        train_fold['flag'] = train_fold['Labels'].apply(lambda x: 1 if str(args.cls_label) in x else 0)
        valid_fold['flag'] = valid_fold['Labels'].apply(lambda x: 1 if str(args.cls_label) in x else 0)
        train_fold = train_fold[train_fold['flag']==1].reset_index(drop=True)
        valid_fold = valid_fold[valid_fold['flag']==1].reset_index(drop=True)
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

    # model_name = args.model if '-' not in args.model else args.model.split('-')[0]
    # if model_name.startswith('effi'):
    #     model_name = model_name[:-2] + '-' + model_name[-2:]
    # #model = model_steel(model_name, pretrained=True, down=False)
    # if model_name == 'resnext101_32x16d':
    #     encoder_weights = 'instagram'
    # else:
    #     encoder_weights = 'imagenet'
    # if args.framework == 'Unet':
    #     model = smp.Unet(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None)
    # elif args.framework == 'FPN':
    #     model = smp.FPN(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None)
    # elif args.framework == 'JPU':
    #     model = model_cloud_JPU(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None)
    # elif '_' in args.framework:
    #     framework = args.framework.split('_')[0]
    #     model = model_cloud_smp(framework, model_name, classes=args.n_classes, pretrained=True)
    # else:
    #     raise RuntimeError('Framework %s not implemented.' % (args.framework))
    model = get_model(args)

    if args.mode == 'train':
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        training_set = Dataset_cloud(train_root,
                                     df=train_fold, transform=train_transform,
                                     mode='train',
                                     cls_label=args.cls_label)
        #sampler = EmptySampler(data_source=training_set, positive_ratio_range=sampler_ratio, epochs=args.n_epochs)
        validation_set = Dataset_cloud(train_root,
                                       df=valid_fold, transform=test_transform,
                                       mode='train',
                                       cls_label=args.cls_label)

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
        elif args.lrc == '':
            scheduler = None
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
                                      mode='test',
                                      cls_label=args.cls_label)
            valid_loader = DataLoader(valid_set, shuffle=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.workers)
            predict(model, args.mode, loader=valid_loader, out_path=run_root, fold=args.fold, tta=args.tta, args=args)

        elif args.mode == 'predict_test':
            if args.limit:
                ss = ss[:args.limit]
            test_set = Dataset_cloud(test_root,
                                     df=ss, transform=test_transform,
                                     mode='test',
                                     cls_label=args.cls_label)
            test_loader = DataLoader(test_set, shuffle=False,
                                     batch_size=args.batch_size,
                                     num_workers=args.workers)
            predict(model, args.mode, loader=test_loader, out_path=run_root, fold=args.fold, tta=args.tta, args=args)
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
                    fold=args.fold, tta=args.tta, args=args,
                    pixel_thresholds=pixel_ts_list,
                    area_size=area_ts_list,
                    )
    else:
        print('%s mode not implemented' % (args.mode))

def get_model(args):
    model_name = args.model if '-' not in args.model else args.model.split('-')[0]
    if model_name.startswith('effi'):
        model_name = model_name[:-2] + '-' + model_name[-2:]
    #model = model_steel(model_name, pretrained=True, down=False)
    if model_name == 'resnext101_32x16d':
        encoder_weights = 'instagram'
    else:
        encoder_weights = 'imagenet'
    if args.framework == 'Unet':
        model = smp.Unet(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None)
    elif args.framework == 'FPN':
        model = smp.FPN(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None)
    elif args.framework == 'JPU':
        model = model_cloud_JPU(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None)
    elif '_' in args.framework:
        framework = args.framework.split('_')[0]
        model = model_cloud_smp(framework, model_name, classes=args.n_classes, pretrained=True)
    else:
        raise RuntimeError('Framework %s not implemented.' % (args.framework))
    return model

def train(args, model: nn.Module, optimizer, scheduler, *,
          train_loader, valid_loader, epoch_length,  # use_cuda,
          n_epochs=None, strategy=None) -> bool:
    n_epochs = n_epochs or args.n_epochs

    run_root = Path(args.run_root)
    model_path = run_root / ('model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    best_dice_path = run_root / ('best-dice-%d.pt' % args.fold)
    best_acc_path = run_root / ('best-acc-%d.pt' % args.fold)

    best_dice_channel = [0, 0, 0, 0]

    if best_dice_path.exists():
        state, best_valid_loss = load_model(model, best_dice_path)
        best_valid_dice = state['best_valid_dice']
        best_valid_loss = state['best_valid_loss']
        if 'best_valid_acc' in state:
            best_valid_acc = state['best_valid_acc']
        else:
            best_valid_acc = 0
        start_epoch = state['epoch'] + 1
        step = state['step']
        best_epoch = state['epoch']
    else:
        best_valid_dice = 0
        best_valid_loss = 999999
        best_valid_acc = 0
        start_epoch = 0
        step = 0
        best_epoch = 0
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_valid_dice': val_dice,
        'best_valid_loss': val_loss,
        'best_valid_acc': val_acc,
    }, str(model_path))

    if args.channel_wise:
        best_dice_path_channel = [run_root / ('best-dice-%d-c%d.pt' % (args.fold, x)) for x in range(4)]
    report_each = 1000000
    if (run_root / ('train_%d.log' % (args.fold))).exists():
        log = run_root.joinpath('train-%d.log' % args.fold).open('at', encoding='utf8')
    else:
        log = run_root.joinpath('train-%d.log' % args.fold).open('at', encoding='utf8')
        log.write('epoch|   lr   |  loss |val loss|bce loss|dice loss|fc loss|val dice|  neg,  pos,pos_1,pos_2,pos_3,pos_4|cls_1,cls_2,cls_3,cls_4| TN | TP | tnr , tpr , acc |time|save\n')
                   #00000|0.000000|0.00000| 0.00000| 0.00000|  0.00000|0.00000| 0.00000|0.000,0.000,0.000,0.000,0.000,0.000|00.0| *

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
            if isinstance(logits, tuple):
                logits, logits_fc = logits
                logits_fc = logits_fc.view(batch_size, channel)
            else:
                logits_fc = logits.view(batch_size, channel, -1)
                logits_fc = torch.max(logits_fc, -1)[0]

            if args.pos_only and args.cls_label not in [0,1,2,3]:
                if args.pos_only == 2:
                    loss, _ = combo_loss_posDice(logits, labels=targets, fc=logits_fc,
                                                 labels_fc=targets_fc, weights=args.loss_weights)
                elif args.pos_only == 1:
                    loss, _ = combo_loss_onlypos(logits, labels=targets, fc=logits_fc,
                                                 labels_fc=targets_fc, weights=args.loss_weights)
            else:
                loss, _ = combo_loss(logits, labels=targets, fc=logits_fc,
                                     labels_fc=targets_fc, weights=args.loss_weights,
                                     per_image = args.loss_per_image)
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
        if args.pos_only==1:
            val_dice = valid_metrics['dice_detail'][1]
        else:
            val_dice = valid_metrics['val_dice']
        val_loss = valid_metrics['val_loss']
        dice_details = valid_metrics['dice_detail']
        dice_by_channel = valid_metrics['dice_by_channel']
        tn = valid_metrics['hit'][0]
        tp = sum(valid_metrics['hit'][1:])
        tnr, tpr, val_acc = valid_metrics['rate']
        _save_ckp = ' '

        save(epoch + 1)

        if epoch < start_epoch + n_epochs - 1:
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_dice)
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

        if val_dice > best_valid_dice:
            shutil.copy(str(model_path), str(best_dice_path))
            best_valid_dice = val_dice
            _save_ckp += '~'
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            #best_valid_dice = val_dice
            shutil.copy(str(model_path), str(best_model_path))
            best_epoch = epoch
            _save_ckp += '*'
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            shutil.copy(str(model_path), str(best_acc_path))
            _save_ckp += '#'
        if args.channel_wise:
            for i in range(4):
                if best_dice_channel[i] < dice_by_channel[i]:
                    best_dice_channel[i] = dice_by_channel[i]
                    shutil.copy(str(model_path), str(best_dice_path_channel[i]))
        run_time = (time.time() - start_time)/60

        # On circle end, save snap, reset metrics
        #if isinstance(scheduler, CosineAnnealingLR):
        #    n_circle = epoch // (args.patience)
        #    if epoch % (args.patience) == 0 and n_circle >= 1:
        #        best_dice_path_snap = run_root / ('best-dice-%d-c%d.pt' % (args.fold, n_circle))
        #        best_cls_path_snap = run_root / ('best-cls-%d-c%d.pt' % (args.fold, n_circle))
        #        shutil.copy(str(best_dice_path), str(best_dice_path_snap))

        #        shutil.copy(str(best_cls_path), str(best_cls_path_snap))
        #        best_valid_dice = valid_dice
        #        best_neg_precision = val_neg_precision
        #        best_valid_loss = valid_loss
        # write log
        #log.write('epoch|   lr   |  loss |val loss|bce loss|dice loss|fc loss|val dice| neg,  pos,pos_1,pos_2,pos_3,pos_4|time|save\n')
                  ##00000|0.000000|0.00000|0.00000| 0.00000|  0.00000|0.00000| 0.00000|0.000,0.000,0.000,0.000,0.000,0.000|00.0| *
        log.write('%5d|%1.6f|%1.5f| %1.5f| %1.5f|  %1.5f|%1.5f| %1.5f|%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f|%1.3f,%1.3f,%1.3f,%1.3f|%4d|%4d|%.3f,%.3f,%.3f|%.2f|%s' \
                  % (epoch, lr, mean_loss, val_loss, valid_metrics['bce_loss'],
                     valid_metrics['dice_loss'], valid_metrics['fc_loss'],
                     val_dice, dice_details[0], dice_details[1], dice_details[2],
                     dice_details[3], dice_details[4], dice_details[5],
                     dice_by_channel[0], dice_by_channel[1], dice_by_channel[2],
                     dice_by_channel[3], tn, tp, tnr, tpr, val_acc, run_time, _save_ckp))

        log.write('\n')
        log.flush()
    print('Best epoch: %d, Loss: %.5f, Dice: %.5f, Dice_channel: %.5f.'\
          % (best_epoch, best_valid_loss, best_valid_dice, np.mean(best_dice_channel)))
    return True

def validation(model: nn.Module, valid_loader, args, save_result=False) -> Dict[str, float]:
    run_root = Path(args.run_root)
    model.eval()
    all_predictions, all_targets, all_dices = [], [], []
    all_targets_cls, all_cls = [], []
    dice_details, sample_details = [], []
    all_losses, bce_losses, dice_losses, fc_losses = 0, 0, 0, 0
    n_samples = 0
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(valid_loader, ascii=True):
            batch_size, channel, _, _ = targets.size()
            all_targets.append(targets.numpy().copy())
            inputs, targets = inputs.cuda(), targets.cuda()
            targets_fc = (targets.view(batch_size, channel, -1).sum(-1)>0).float()

            if args.sliding:
                logits, logits_fc = predict_sliding(model, inputs)
            else:
                logits = model(inputs)

            if isinstance(logits, tuple):
                logits, logits_fc = logits
                logits_fc = logits_fc.view(batch_size, channel)
            else:
                logits_fc = logits.view(batch_size, channel, -1)
                logits_fc = torch.max(logits_fc, -1)[0]

            if args.pos_only and args.cls_label not in [0,1,2,3]:
                if args.pos_only == 2:
                    loss, loss_details = combo_loss_posDice(logits, labels=targets, fc=logits_fc,
                                                            labels_fc=targets_fc, weights=args.loss_weights)
                elif args.pos_only == 1:
                    loss, loss_details = combo_loss_onlypos(logits, labels=targets, fc=logits_fc,
                                                            labels_fc=targets_fc, weights=args.loss_weights)
            else:
                loss, loss_details = combo_loss(logits, labels=targets, fc=logits_fc,
                                                labels_fc=targets_fc, weights=args.loss_weights,
                                                per_image = args.loss_per_image)
            loss_seg_bce, loss_seg_dice, loss_fc = loss_details

            all_losses += loss.item()*batch_size
            bce_losses += loss_seg_bce.item()*batch_size
            dice_losses += loss_seg_dice.item()*batch_size
            fc_losses += loss_fc.item()*batch_size
            n_samples += batch_size

            probs = F.sigmoid(logits)
            dice_by_channel, dice_detail, sample_detail = dice_channel_torch(probs, targets)
            dice_detail = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in dice_detail]
            dice_details.append(dice_detail)
            dice_by_channel = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in dice_by_channel]
            all_dices.append(dice_by_channel)
            sample_details.append(sample_detail)
            all_cls.append(F.sigmoid(logits_fc).data.cpu().numpy())
            all_targets_cls.append(targets_fc.data.cpu().numpy())

    all_dices = np.sum(all_dices, axis=0)/n_samples
    metrics = {}
    metrics['val_loss'] = float(all_losses/n_samples)
    metrics['dice_loss'] = float(dice_losses/n_samples)
    metrics['bce_loss'] = float(bce_losses/n_samples)
    metrics['fc_loss'] = float(fc_losses/n_samples)
    metrics['val_dice'] = float(np.mean(all_dices))
    metrics['dice_by_channel'] = all_dices
    dice_details = np.sum(dice_details,axis=0)
    sample_details = np.sum(sample_details,axis=0)
    metrics['dice_detail'] = [float(x)/float(y+1e-9) for x, y in zip(dice_details, sample_details)]
    all_cls, all_targets_cls = np.concatenate(all_cls), np.concatenate(all_targets_cls)

    metrics['hit'], _, _ = metric_hit(all_cls, all_targets_cls)
    tnr = metrics['hit'][0] / np.sum(all_targets_cls==0)
    tpr = sum(metrics['hit'][1:]) / np.sum(all_targets_cls==1)
    acc = sum(metrics['hit'])/all_cls.shape[0]/4
    metrics['rate'] = [tnr, tpr, acc]

    to_print = []
    for idx, (k, v) in enumerate(metrics.items()):
        if k == 'dice_by_channel':
            v_p = ['%.3f' % (x) for x in v]
            v_p = ','.join(v_p)
            to_print.append(f'{k} [{v_p}]')
        elif k == 'hit' or k == 'dice_detail' or k == 'rate':
            pass
        else:
            to_print.append(f'{k} {v:.3f}')
    #to_print.append(str(np.sum(pred)))
    print(' | '.join(to_print))
    return metrics

def predict_1batch(model, inputs, tta):
    inputs = inputs.cuda()
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs, logits_fc = outputs
        B, C, H, W = outputs.shape
    else:
        B, C, H, W = outputs.shape
        logits_fc = outputs.view(B, C, -1)
        logits_fc = torch.max(logits_fc, -1)[0]
    logits_fc = logits_fc.view(B, C)

    cls_pred = F.sigmoid(logits_fc).view(-1).data.cpu().numpy()
    mask_pred = F.sigmoid(outputs).view(-1, H, W).data.cpu().numpy()

    if tta >= 2:# h flip
        # mask
        outputs = model(inputs.flip(3))
        if isinstance(outputs, tuple):
            outputs, logits_fc = outputs
        else:
            logits_fc = outputs.view(B, C, -1)
            logits_fc = torch.max(logits_fc, -1)[0]
        logits_fc = logits_fc.view(B, C)

        outputs = outputs.flip(3)
        mask_pred += F.sigmoid(outputs).view(-1, H, W).data.cpu().numpy()
        # cls
        cls_pred += F.sigmoid(logits_fc).view(-1).data.cpu().numpy()
    if tta == 3:# v flip
        # mask
        outputs = model(inputs.flip(2))
        if isinstance(outputs, tuple):
            outputs, logits_fc = outputs
        else:
            logits_fc = outputs.view(B, C, -1)
            logits_fc = torch.max(logits_fc, -1)[0]
        logits_fc = logits_fc.view(B, C)
        outputs = outputs.flip(2)
        mask_pred += F.sigmoid(outputs).view(-1, H, W).data.cpu().numpy()
        # cls
        logits_fc = outputs.view(B, C, -1)
        logits_fc = torch.max(logits_fc, -1)[0]
        cls_pred += F.sigmoid(logits_fc).view(-1).data.cpu().numpy()

    if tta != 0:
        cls_pred /= tta
        mask_pred /= tta
    return mask_pred, cls_pred

def predict(model, mode, loader, out_path: Path, fold, tta, args,
            pixel_thresholds=PIXEL_THRESHOLDS,
            area_size=AREA_SIZES):
    mode = mode.split('_')[-1]
    model.eval()
    all_outputs, all_ids, all_cls = [], [], []
    df_sub = []
    with torch.no_grad():
        for inputs, names in tqdm.tqdm(loader, desc='Predict'):
            mask_pred, cls_pred = predict_1batch(model, inputs, tta)
            all_cls.append(cls_pred)

            ids = [item for sublist in list(zip(*names)) for item in sublist]
            df_batch = build_sub(ids, mask_pred, mode, thresholds=pixel_thresholds,
                                 area_size=area_size)
            df_sub.append(df_batch)
            all_ids.extend(ids)

    df = pd.DataFrame(data=np.concatenate(all_cls), index=all_ids)
    df = mean_df(df).reset_index()
    df.rename(columns={'index': 'Image_Label'}, inplace=True)
    df['EncodedPixels'] = np.nan
    df['EncodedPixels'].loc[df[0] > 0.5] = '1 1'

    df_sub = pd.concat(df_sub)

    if mode == 'opt':
        df.to_csv(out_path / ('%s_cls_fold%d_tta%d.csv' % (mode, fold, tta)), index=None)
        df_sub.to_csv(out_path / ('%s_fold%d_tta%d_[%s].csv' % (mode, fold, tta, str(pixel_thresholds[0]))), index=None)
    else:
        if tta <= 1:
            df.to_csv(out_path / ('%s_cls_fold%d.csv' % (mode, fold)), index=None)
            df_sub.to_csv(out_path / ('%s_fold%d.csv' % (mode, fold)), index=None)
        else:
            df.to_csv(out_path / ('%s_cls_fold%d_tta%d.csv' % (mode, fold, tta)), index=None)
            df_sub.to_csv(out_path / ('%s_fold%d_tta%d.csv' % (mode, fold, tta)), index=None)
    print(f'Saved predictions to {out_path}')

def predict_5fold(loader, out_path: Path, args,
                  pixel_thresholds=PIXEL_THRESHOLDS,
                  area_size=AREA_SIZES):
    tta = args.tta
    run_root = Path(args.run_root)

    # model_name = args.model if '-' not in args.model else args.model.split('-')[0]
    # if model_name.startswith('effi'):
    #     model_name = model_name[:-2] + '-' + model_name[-2:]
    # if model_name == 'resnext101_32x16d':
    #     encoder_weights = 'instagram'
    # else:
    #     encoder_weights = 'imagenet'
    # if args.framework == 'Unet':
    #     models = [smp.Unet(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None) for x in range(5)]
    # elif args.framework == 'FPN':
    #     models = [smp.FPN(model_name, classes=args.n_classes, encoder_weights=encoder_weights, activation=None) for x in range(5)]
    # elif '_' in args.framework:
    #     framework = args.framework.split('_')[0]
    #     models = [model_cloud_smp(framework, model_name, classes=args.n_classes, pretrained=True) for x in range(5)]
    # else:
    #     raise RuntimeError('Framework %s not implemented.' % (args.framework))
    models = [get_model(args) for x in range(5)]

    for fold in range(NFOLDS):
        if (run_root /('best-dice-%d.pt' % fold)).exists():
            load_model(models[fold], run_root /('best-dice-%d.pt' % fold), multi2single=False)
        else:
            load_model(models[fold], run_root /('best-model-%d.pt' % fold), multi2single=False)
        models[fold] = models[fold].cuda()
        models[fold].eval()

    all_outputs, all_ids, all_cls = [], [], []
    df_sub = []
    with torch.no_grad():
        for inputs, names in tqdm.tqdm(loader, desc='Predict'):
            for fold in range(NFOLDS):
                if fold == 0:
                    mask_pred, cls_pred = predict_1batch(models[fold], inputs, tta)
                else:
                    m1, c1 = predict_1batch(models[fold], inputs, tta)
                    mask_pred += m1
                    cls_pred += c1
            mask_pred /= NFOLDS
            cls_pred /= NFOLDS
            all_cls.append(cls_pred)
            ids = [item for sublist in list(zip(*names)) for item in sublist]
            df_batch = build_sub(ids, mask_pred, args.mode, thresholds=pixel_thresholds,
                                 area_size=area_size)
            df_sub.append(df_batch)
            all_ids.extend(ids)

    df = pd.DataFrame(data=np.concatenate(all_cls), index=all_ids)
    df = mean_df(df).reset_index()
    df.rename(columns={'index': 'Image_Label'}, inplace=True)
    df['EncodedPixels'] = np.nan
    df['EncodedPixels'].loc[df[0] > 0.5] = '1 1'

    df_sub = pd.concat(df_sub)

    df.to_csv(out_path / ('test_cls_%dfold_tta%d.csv' % (NFOLDS, tta)), index=None)
    df_sub.to_csv(out_path / ('test_%dfold_tta%d.csv' % (NFOLDS, tta)), index=None)

    print(f'Saved predictions to {out_path}')

if __name__ == '__main__':
    main()
