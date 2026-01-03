import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from datasets.cifar_dataloader import CIFAR100_Cls
from models.models import classifier32, classifier32ABN
from models.resnet import resnet

from utils import save_networks, load_networks
from core_acc import train, test_cls


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cifar100', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--feat_dim', type=int, default=128, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=250)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)
parser.add_argument('--centroid-path', type=str, default='./Estimated_prototypes/100centers_128dim.pth')

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='resnet32')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)


def main_worker(options):
    print("options: {}".format(options))

    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']:
        use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'cifar100' == options['dataset']:
        Data = CIFAR100_Cls(dataroot=options['dataroot'], batch_size=options['batch_size'],
                            img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
    else:
        raise NotImplementedError("Only CIFAR100 has been kept in this simplified version.")

    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    if options['model'] == 'resnet32':
        net = resnet(depth=32, output_dims=128, num_class=100, multiplier=1)
    elif options['model'] == 'resnet20':
        net = resnet(depth=32, output_dims=128, num_class=10, multiplier=1)

    else:
        print("None Net")





    options.update({'use_gpu': use_gpu})
    Loss = importlib.import_module('loss.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if options['dataset'] == 'cifar100':
        model_path += '_full'
        file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['cs'])
    else:
        file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['cs'])

    # ----------------------
    
    acc_list = []
    # ----------------------

    # Evaluation only
    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test_cls(net, criterion, testloader, epoch=0, **options)
        print("Acc (%): {:.3f}".format(results['ACC']))
        return results

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]

    if options['dataset'] == 'tiny_imagenet':
        optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[90, 120, 150,210])

    start_time = time.time()

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        
        # if epoch % 10 == 0:
        #     criterion.Dist.update_matching()

        if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test_cls(net, criterion, testloader, epoch=epoch, **options)
            acc = results['ACC']
            acc_list.append(acc)  
            print("Acc (%): {:.3f}".format(acc))
            save_networks(net, model_path, file_name, criterion=criterion)

        if options['stepsize'] > 0:
            scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    # ----------------------

    best_acc = max(acc_list) if len(acc_list) > 0 else 0
    results['best_ACC'] = best_acc
    print("Best Acc (%): {:.3f}".format(best_acc))
    # ----------------------

    return results, acc_list



if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)

    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    options['img_size'] = img_size
    results = dict()

    
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")

    if options['dataset'] == 'cifar100':
        file_name = '{}_{}_full.csv'.format(options['dataset'], time_str)
    else:
        file_name = '{}_{}.csv'.format(options['dataset'], time_str)

    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join(options['outf'], 'results', dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    
    res, acc_list = main_worker(options)
    results['final'] = res
    results['best_ACC'] = res['best_ACC']

    # DataFrame 1: （final + best）
    df1 = pd.DataFrame([results])

    # DataFrame 2: 
    df2 = pd.DataFrame({
        'epoch': list(range(1, len(acc_list) + 1)),
        'ACC': acc_list
    })

    
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    df.to_csv(os.path.join(dir_path, file_name), index=False)



