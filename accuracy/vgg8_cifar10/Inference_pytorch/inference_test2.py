import argparse
import os
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
from utee import hook
#from IPython import embed
from datetime import datetime
from subprocess import call
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|ResNet18')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='140,180', help='decreasing strategy')
parser.add_argument('--wl_weight', type=int, default=8)
parser.add_argument('--wl_grad', type=int, default=8)
parser.add_argument('--wl_activate', type=int, default=8)
parser.add_argument('--wl_error', type=int, default=8)
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', type=int, default=1, help='run hardware inference simulation')
parser.add_argument('--subArray', type=int, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--parallelRead', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
parser.add_argument('--ADCprecision', type=int, default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', type=float, default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', type=float, default=0., help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
	logger('{}: {}'.format(k, v))
logger("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

# data loader and model
assert args.dataset in ['cifar10', 'cifar100', 'imagenet'], args.dataset
if args.dataset == 'cifar10':
    train_loader, test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'cifar100':
    train_loader, test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1)
elif args.dataset == 'imagenet':
    train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
else:
    raise ValueError("Unknown dataset type")
    
assert args.model in ['VGG8', 'DenseNet40', 'ResNet18'], args.model
if args.model == 'VGG8':
    from models import VGG
    #model_path = "/home/ssdl_neurosim1/All_Neurosim_Project/VGG8_CIFAR10/Inference_pytorch/log/default/ADCprecision=5/batch_size=64/cellBit=4/dataset=cifar10/decreasing_lr=140,180/detect=0/grad_scale=8/inference=0/lr=0.01/mode=WAGE/model=VGG8/onoffratio=10/parallelRead=128/seed=117/subArray=128/t=0/target=0/v=0/vari=0/wl_activate=8/wl_error=8/wl_grad=8/wl_weight=2/best-154.pth"
    model_path = './log/VGG8.pth'
    modelCF = VGG.vgg8(args = args, logger=logger, pretrained = model_path)
elif args.model == 'DenseNet40':
    from models import DenseNet
    model_path = './log/DenseNet40.pth'     # WAGE mode pretrained model
    modelCF = DenseNet.densenet40(args = args, logger=logger, pretrained = model_path)
elif args.model == 'ResNet18':
    from models import ResNet
    # FP mode pretrained model, loaded from 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    # model_path = './log/xxx.pth'
    # modelCF = ResNet.resnet18(args = args, logger=logger, pretrained = model_path)
    modelCF = ResNet.resnet18(args = args, logger=logger, pretrained = True)
else:
    raise ValueError("Unknown model type")

print("model_path", model_path)

w = next(p for n,p in modelCF.named_parameters() if "weight" in n)
print("weight abs max:", w.abs().max().item())

if args.cuda:
	modelCF.cuda()

best_acc, old_file = 0, None
t_begin = time.time()

# ready to go
modelCF.eval()

test_loss = 0
correct = 0
trained_with_quantization = True

criterion = torch.nn.CrossEntropyLoss()

modelCF.eval()
test_loss = 0.0
correct = 0
total = 0
seen_batches = 0

hook_handle_list = None

for i, (data, target) in enumerate(test_loader):
    if args.cuda:
        data, target = data.cuda(), target.cuda()

    with torch.no_grad():
        output = modelCF(data)

        if i == 0:   # ✅ shape는 한 번만
            print("output shape:", output.shape)
            print("data stats:", data.min().item(), data.max().item(), data.mean().item(), data.std().item())
            print("first 10 classes:", test_loader.dataset.classes[:10])
            print("class_to_idx sample:", list(test_loader.dataset.class_to_idx.items())[:5])
            print("first logits (top5):", torch.topk(output[0].detach().cpu(), 5))        
            out = modelCF.features[0](data)  # VGG 구현에 따라 접근은 달라질 수 있음
            print(out.min().item(), out.max().item(), out.mean().item(), out.std().item())



        loss = criterion(output, target)
        test_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        seen_batches += 1


        # #(디버깅용) 몇 배치만 돌리고 싶으면 여기서 break
        if i == 0:
            break

avg_loss = test_loss / seen_batches
acc = 100.0 * correct / total
logger(f"Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)")
