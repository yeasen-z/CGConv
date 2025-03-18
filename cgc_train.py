import os, sys, shutil, time, random
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import models

import torch.nn.functional as F
import random
import pandas as pd
import numpy as np

import models

from cgToolkit import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# ------------------------------------- settinh arges ------------------------------------- #
parser = argparse.ArgumentParser(
    description='Training network for image classification',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/elliot/data/pytorch/svhn/',
                    type=str,
                    help='Path to dataset')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['cifar10', 'mnist'],
    help='Choose datasets')

parser.add_argument('--arch',
                    metavar='ARCH',
                    default='lbcnn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnext29_8_64)')

parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='Number of epochs to train.')
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay',
                    type=float,
                    default=1e-4,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument(
    '--gammas',
    type=float,
    nargs='+',
    default=[0.1, 0.1],
    help=
    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule'
)

parser.add_argument('--print_freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 200)')
parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument(
    '--fine_tune',
    dest='fine_tune',
    action='store_true',
    help='fine tuning from the pre-trained model, force the start epoch be zero'
)
parser.add_argument('--model_only',
                    dest='model_only',
                    action='store_true',
                    help='only save the model without external utils_')

parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id',
                    type=int,
                    default=0,
                    help='device range [0,ngpu-1]')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')

parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

parser.add_argument(
    '--quan_bitwidth',
    type=int,
    default=None,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')

parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=None,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')

# Piecewise clustering
parser.add_argument('--clustering',
                    dest='clustering',
                    action='store_true',
                    help='add the piecewise clustering term.')
parser.add_argument('--lambda_coeff',
                    type=float,
                    default=1e-3,
                    help='lambda coefficient to control the clustering term')

##########################################################################

parser.add_argument('--bfa_mydefense',
                    dest='enable_bfa_mydefense',
                    action='store_true',
                    help='enable our defense')
parser.add_argument('--ic_only',
                    dest='ic_only',
                    default=False,
                    #action='store_false',
                    help='enable ic only trainning')
                    

parser.add_argument('--weight', default='1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1', 
                      help='weight')

parser.add_argument('--sponge_sigma', default=1e-04, type=float, help='Correction constant')
parser.add_argument('--sponge_lb', default=0.12, type=float, help='Sponge penalty')
parser.add_argument('--sponge_criterion', default='l0', type=str, help='Sponge penalty')
parser.add_argument('--cgc', action='store_true', help='if true, the model will be trained with centre-zero awareness')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = 0 #random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

###############################################################################
###############################################################################

min_pos=64
max_neg=-64

def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log',
                           'run_' + str(args.manualSeed))
    # logger = Logger(tb_path)
    writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)
    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name 
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            all_param),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15] , gamma=0.1)

        

    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(
            filter(lambda param: param.requires_grad, net.parameters()),
            lr=state['learning_rate'],
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            #checkpoint = torch.load(args.resume)
            checkpoint = torch.load('/data/workplace/zms/ICCV25/Train_models/save/cifar10_cgc_resnet56_500_SGD_resnet56_cgc_333/model_best.pth.tar',weights_only=False)#checkpoint_branch.pth.tar#model_best_def.pth.tar
            #checkpoint = torch.load('/home/wangjialai/copy_for_use/flip_attack/BFA_sdn_branch/save/cifar10_vgg16_quan_200_SGD_binarized/checkpoint_branch.pth.tar')#checkpoint_branch.pth.tar#model_best_def.pth.tar
            
            #if not (args.fine_tune):
            if True:
                args.start_epoch = 0#checkpoint['epoch']
                recorder = checkpoint['recorder']
                #optimizer.load_state_dict(checkpoint['optimizer'])

            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)
            
            #net.load_state_dict(state_tmp)
            model_dict = net.state_dict()
            pretrained_dict = {k:v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)


            print_log(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume),
                      log)
    elif args.enable_bfa_mydefense:
        checkpoint = torch.load("/home/wangjialai/copy_for_use/flip_attack/BFA/save/cifar10_vgg16_quan_160_SGD_binarized/model_best.pth.tar")
        state_tmp = net.state_dict()
        if 'state_dict' in checkpoint.keys():
            state_tmp.update(checkpoint['state_dict'])
        else:
            state_tmp.update(checkpoint)

        net.load_state_dict(state_tmp)
            

    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)
        
    val_acc, _, val_los = validate(test_loader, net, criterion, log)
    #sys.exit()
    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()

    # block for weight reset
    
    if args.reset_weight:
        print("reset weight")
        for name, m in net.named_modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                m.__reset_weight__()
            
   

    list_shape = []
    for name, m in net.named_modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            #if m.weight.grad is not None:
            if ('conv_1' in name) or ('stage_1' in name):
                num = m.weight.flatten().__len__()
                zero_n = np.zeros(num)
                zero_n[:20]=1
                np.random.shuffle(zero_n)
                zero_r=zero_n.reshape(m.weight.shape)
                torch_m=torch.tensor(zero_r)
                list_shape.append(torch_m)
                


    val_acc, _, val_los = validate(test_loader, net, criterion, log)
    

    if args.evaluate:
        print("first evaluate:")
        _,_,_, output_summary = validate(test_loader, net, criterion, log)
        pd.DataFrame(output_summary).to_csv(os.path.join(args.save_path, 'output_summary_{}.csv'.format(args.arch)),
                                            header=['top-1 output'], index=False)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    
    val_acc, _, val_los = validate(test_loader, net, criterion, log)
    print("epoch :", args.start_epoch, args.epochs)
    count=0
    is_best_defense_best=0
    is_best_defense=0
    if args.resume:
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad,
                                            net.parameters()),
                                     lr=0.01,
                                     weight_decay=0.0005)
    for epoch in range(args.start_epoch, args.epochs):
        count+=1

        current_learning_rate, current_momentum = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer,
                                     epoch, log)


        # evaluate on validation set
        val_acc, _, val_los = validate(test_loader, net, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)
        
        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }
        #is_best=True
        save_checkpoint(checkpoint_state, is_best, args.save_path,
                        'checkpoint.pth.tar', log)


        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#

        ## Log the graidents distribution
        for name, param in net.named_parameters():
            name = name.replace('.', '/')
            try:
                writer.add_histogram(name + '/grad',
                                    param.grad.clone().cpu().data.numpy(),
                                    epoch + 1,
                                    bins='tensorflow')
            except:
                pass
            
            try:
                writer.add_histogram(name, param.clone().cpu().data.numpy(),
                                      epoch + 1, bins='tensorflow')
            except:
                pass
            
        total_weight_change = 0 
            
        for name, module in net.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                try:
                    writer.add_histogram(name+'/bin_weight', module.bin_weight.clone().cpu().data.numpy(), epoch + 1,
                                        bins='tensorflow')
                    writer.add_scalar(name + '/bin_weight_change', module.bin_weight_change, epoch+1)
                    total_weight_change += module.bin_weight_change
                    writer.add_scalar(name + '/bin_weight_change_ratio', module.bin_weight_change_ratio, epoch+1)
                except:
                    pass
                
        writer.add_scalar('total_weight_change', total_weight_change, epoch + 1)
        print('total weight changes:', total_weight_change)

        writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)

    
    """ if args.enable_bfa_mydefense:# load our defense model
        checkpoint = torch.load(args.resume)
        #checkpoint = torch.load('./save/cifar10_resnet20_quan_BFA_defense_test/model_best.pth.tar')
    #if not (args.fine_tune):
    if True:
        args.start_epoch = checkpoint['epoch']
        recorder = checkpoint['recorder']
        optimizer.load_state_dict(checkpoint['optimizer'])

    state_tmp = net.state_dict()
    if 'state_dict' in checkpoint.keys():
        state_tmp.update(checkpoint['state_dict'])
    else:
        state_tmp.update(checkpoint)

    net.load_state_dict(state_tmp) """
    
    # val_acc, _, val_los = validate(test_loader, net, criterion, log)
        
    log.close()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    ncfg=[3,3,3]
    if args.ngpu > 1:
        victim_leaf_nodes=[module for module in model.module.stage_1.modules() if isinstance(module, nn.ReLU)][ncfg[0]*2:]
        victim_leaf_nodes+=[module for module in model.module.stage_2.modules() if isinstance(module, nn.ReLU)][ncfg[1]*2:]
        victim_leaf_nodes+=[module for module in model.module.stage_3.modules() if isinstance(module, nn.ReLU)][ncfg[2]*2:]
    else:
        victim_leaf_nodes=[module for module in model.stage_1.modules() if isinstance(module, nn.ReLU)][ncfg[0]*2:]
        victim_leaf_nodes+=[module for module in model.stage_2.modules() if isinstance(module, nn.ReLU)][ncfg[1]*2:]
        victim_leaf_nodes+=[module for module in model.stage_3.modules() if isinstance(module, nn.ReLU)][ncfg[2]*2:]
        
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda()  # the copy will be asynchronous with respect to the host.
            input = input.cuda()


        # compute output
        output = model(input)
        loss = criterion(output, target)
        
        if args.clustering:
            loss += clustering_loss(model, args.lambda_coeff)
        
        if args.cgc:
            sp_loss, sp_stats=compute_sponge_loss(model,input,victim_leaf_nodes,args)
            loss+=sp_loss

        loss.backward()
          
        optimizer.step()
        optimizer.zero_grad()
        loss2 = loss 
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss2.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # clear all grad for model
        for param in model.parameters():
            param.grad = None

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            print_log(
                '  Epoch: [{:03d}][{:03d}/{:03d}]   '
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                'Prec_Bmain@1 {top1_main.val:.3f} ({top1_main.avg:.3f})   '
                'Prec_Bmain@5 {top5_main.val:.3f} ({top5_main.avg:.3f})   '
                .format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1_main=top1,
                    top5_main=top5,
                    
                    ) + time_string(), log)
    print_log(
        '  **Train** Prec_Bmain@1 {top1_bmain.avg:.3f} Prec_Bmain@5 {top5_bmain.avg:.3f} Error@main {errormain:.3f}'
        .format(
                top1_bmain=top1, top5_bmain=top5, errormain=100 - top1.avg,
        ), log)
    
    
    
    top1_avg = top1.avg
    return top1_avg, losses.avg


def validate(val_loader, model, criterion, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()



    # switch to evaluate mode
    model.eval()
    output_summary = [] # init a list for output summary

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            # w = list(map(float, args.weight.split(',')))
            output = model(input)
            #loss = criterion(output, target)
            # loss = 0
            # for idx in range(len(output_branch)):
            loss = criterion(output, target)


            # summary the output
            if summary_output:
                tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                output_summary.append(tmp_list)


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            # for idx in range(len(output_branch)):
            #     prec1, prec5 = accuracy(output_branch[idx].data, target, topk=(1, 5))
            #     top1_list[idx].update(prec1, input.size(0)) 
            #     top5_list[idx].update(prec5, input.size(0))

            
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

       
        
        print_log(
        '  **Test** Prec_Bmain@1 {top1_main.avg:.3f} Prec_Bmain@5 {top5_main.avg:.3f} Error@1 {errormain:.3f}'
        .format(
                top1_main=top1, top5_main=top5, errormain=100 - top1.avg,
        ), log)
        
        if summary_output:
            output_summary = np.asarray(output_summary).flatten()
            return top1.avg, top5.avg, losses.avg, output_summary
        
        return top1.avg, top5.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        print("current model is best:", bestname)
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)

def save_checkpoint_def(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best_def.pth.tar')
        print("best def name:", bestname)
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0)
            
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write(
            '{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
