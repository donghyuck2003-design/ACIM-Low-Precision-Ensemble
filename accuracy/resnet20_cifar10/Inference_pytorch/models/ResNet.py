import torch
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d, QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
from utee import wage_quantizer  # WAGEQuantizer_f 사용
import modules.quantization_cpu_np_infer as qmod

try:
    # (구버전 torchvision)
    from torchvision.models.utils import load_state_dict_from_url
except Exception:
    # (신버전 torchvision)
    from torch.hub import load_state_dict_from_url
    
name=0

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

import json
import random

# ---- activation calibration storage ----
_ACT_CALIB = {
    "enabled": False,       # 캘리브레이션 모드인지
    "p": 0.999,             # P99.9
    "max_samples": 200000,  # 레이어별 샘플 최대 저장 (메모리 제한)
    "stride": 16,           # 샘플링 간격(다운샘플)
    "samples": {},          # tag -> list[float]
    "alphas": {},           # tag -> float (캘리브레이션 결과)
}

def calib_enable(p=0.999, max_samples=200000, stride=16):
    _ACT_CALIB["enabled"] = True
    _ACT_CALIB["p"] = float(p)
    _ACT_CALIB["max_samples"] = int(max_samples)
    _ACT_CALIB["stride"] = int(stride)

def calib_disable():
    _ACT_CALIB["enabled"] = False

def calib_update(tag: str, x: torch.Tensor):
    # ReLU 이후 전제: 음수는 0으로 취급
    if x is None:
        return
    with torch.no_grad():
        t = x.detach()
        # 안전하게 ReLU 영역만 본다
        t = t.clamp_min(0.0)

        # 너무 크면 다운샘플 (메모리 방지)
        flat = t.reshape(-1)
        if _ACT_CALIB["stride"] > 1:
            flat = flat[::_ACT_CALIB["stride"]]

        # CPU로 가져와서 리스트에 누적
        vals = flat.float().cpu().tolist()

        buf = _ACT_CALIB["samples"].setdefault(tag, [])
        # 용량 제한
        remaining = _ACT_CALIB["max_samples"] - len(buf)
        if remaining <= 0:
            return
        buf.extend(vals[:remaining])

def calib_finalize():
    """
    저장된 샘플로 P99.9 계산해 alphas 채움.
    """
    alphas = {}
    p = _ACT_CALIB["p"]
    for tag, buf in _ACT_CALIB["samples"].items():
        if len(buf) == 0:
            continue
        v = torch.tensor(buf, dtype=torch.float32)
        a = float(torch.quantile(v, p).item())
        # alpha=0 방지
        if a <= 0:
            a = float(v.max().item())
        alphas[tag] = a
    _ACT_CALIB["alphas"] = alphas
    return alphas

def calib_save_json(path: str):
    with open(path, "w") as f:
        json.dump(_ACT_CALIB["alphas"], f, indent=2)

def calib_load_json(path: str):
    with open(path, "r") as f:
        d = json.load(f)
    d = {str(k): float(v) for k, v in d.items()}
    _ACT_CALIB["alphas"] = d
    return d

def calib_get_alpha(tag: str):
    return _ACT_CALIB["alphas"].get(tag, None)


def act_q(x, args, tag=""):
    if args is None:
        return x

    # (0) calibration 모드면 통계만
    if getattr(args, "act_calib", False) or getattr(qmod, "_ACT_CALIB", {}).get("enabled", False):
        calib_update(tag, x)
        return x

    # (1) HW bit-serial inference에서는 값은 건드리지 말고 alpha_key만 세팅
    #     -> QConv2d/QLinear 안에서 input을 uint-code로 만들며 quantize할 것임(이중 quant 방지)
    if int(getattr(args, "inference", 0)) == 1 and qmod._use_bitserial_hw_path(getattr(args, "model", None)):
        qmod.set_next_alpha_key(tag)   # 다음 QConv/QLinear가 이 key를 peek해서 씀
        if qmod.calib_get_alpha(tag) is None:
            #print("[WARN]", tag, "alpha missing in alpha_dict (tag mismatch?)")
            return x
        return x

    bitsA = int(getattr(args, "wl_activate", -1))
    if bitsA == -1 or bitsA >= 16:
        return x

    # (2) 그 외(비-HW 경로)에서는 기존처럼 alpha-aware quantize 수행
    alpha = qmod.calib_get_alpha(tag)
    if alpha is None:
        #print("[WARN]", tag, "alpha is None")
        return x
    
    # alpha_r = calib_get_alpha(tag)          # ResNet.py store
    # alpha_q = qmod.calib_get_alpha(tag)     # qmod store
    # print("[act_q]", tag, "alpha_resnet", alpha_r, "alpha_qmod", alpha_q)

    x = x.clamp_min(0.0)

    qmod.set_next_alpha_key(tag)
    return wage_quantizer.WAGEQuantizer_f_unsigned_alpha(x, bitsA, alpha, ste=True)





def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,args=None,logger=None):
    """3x3 convolution with padding"""
    global name
    if args.mode == "WAGE":       
        conv2d = QConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, 
                     logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                     wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                     subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                     name = 'Conv3x3'+'_'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
        
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                     subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                     name = 'Conv3x3'+'_'+str(name)+'_' )
    name += 1
    return conv2d


def conv1x1(in_planes, out_planes, stride=1,args=None,logger=None):
    """1x1 convolution"""
    global name
    if args.mode == "WAGE":        
        conv2d = QConv2d(in_planes, out_planes, kernel_size=1, stride=stride, logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                         wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                         name = 'Conv1x1'+'_'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                         logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                         name = 'Conv1x1'+'_'+str(name)+'_' )
    name += 1
    return conv2d


def Conv2d(in_planes, out_planes, kernel_size, stride, padding, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":        
        conv2d = QConv2d(in_planes, out_planes, kernel_size, stride, padding, logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                         wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                         name = 'Conv'+'_'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
    elif args.mode == "FP":
        conv2d = FConv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False,
                         logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                         name = 'Conv'+'_'+str(name)+'_' )
    name += 1
    return conv2d


def Linear(in_planes, out_planes, args=None, logger=None):
    """convolution"""
    global name
    if args.mode == "WAGE":
        linear = QLinear(in_planes, out_planes, 
                        logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error, bias=False,
                        wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                        subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, 
                        name='FC'+'_'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
    elif args.mode == "FP":
        linear = FLinear(in_planes, out_planes, bias=False,
                         logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                         subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, cuda=args.cuda,
                         name='FC'+'_'+str(name)+'_')
    name += 1
    return linear
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, args=None, logger=None, block_tag=""):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,args=args,logger=logger)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,args=args,logger=logger)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.args = args
        self.block_tag = block_tag
        self.name = f"BasicBlock_{id(self)}"  

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = act_q(out, self.args, tag=self.block_tag + "_bb_relu1")

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x_ds = act_q(x, self.args, tag=self.block_tag + "_bb_ds_in")
            # identity = self.downsample[0](x_ds)
            # identity = self.downsample[1](identity)

            if isinstance(self.downsample, nn.Sequential):
                identity = x_ds
                for m in self.downsample:
                    identity = m(identity)
            else:
                identity = self.downsample(x_ds)

        out += identity
        out = self.relu(out)
        #out = act_q(out, self.args, tag="bb_out")
        out = act_q(out, self.args, tag=self.block_tag + "_bb_out")
        
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, args=None, logger=None, block_tag=""):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width,args=args,logger=logger)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation,args=args,logger=logger)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion,args=args,logger=logger)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.args = args
        self.block_tag = block_tag
        self.name = f"BottleneckBlock_{id(self)}"  
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = act_q(out, self.args, tag=self.block_tag + "_bn_relu1")

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = act_q(out, self.args, tag=self.block_tag + "_bn_relu2")

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.downsample is not None:
            if isinstance(self.downsample, nn.Sequential):
                x_ds = act_q(x, self.args, tag=self.block_tag + "_bn_ds_in")
                identity = self.downsample[0](x_ds)
                for m in self.downsample[1:]:
                    identity = m(identity)
            else:
                # 혹시 커스텀 downsample이면, 그 내부에서 conv 직전에 alpha_key 세팅 필요
                identity = self.downsample(x)
                
        out += identity
        out = self.relu(out)
        out = act_q(out, self.args, tag=self.block_tag + "_bn_out")
        #out = act_q(out, self.args, tag="bn_out")

        return out



# models/ResNet.py 수정 부분

class ResNet(nn.Module):
    def __init__(self, block, layers, args, logger, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, deep_stem=False): # deep_stem 인자 추가
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.deep_stem = deep_stem # 플래그 저장
        self.args = args
        self.logger = logger

        # ---- Stem 부분 수정 ----
        if deep_stem:
            # timm resnet10t 스타일의 3단 Stem (24 -> 32 -> 64)
            self.stem = nn.Sequential(
                Conv2d(3, 24, kernel_size=3, stride=2, padding=1, args=args, logger=logger),
                norm_layer(24),
                nn.ReLU(inplace=True),
                Conv2d(24, 32, kernel_size=3, stride=1, padding=1, args=args, logger=logger),
                norm_layer(32),
                nn.ReLU(inplace=True),
                Conv2d(32, 64, kernel_size=3, stride=1, padding=1, args=args, logger=logger),
                norm_layer(64),
                nn.ReLU(inplace=True),
            )
        else:
            # 표준 ResNet 스타일 (7x7)
            self.conv1 = Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, args=args, logger=logger)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 레이어 구성 [1, 1, 1, 1]
        self.layer1 = self._make_layer(block, 64, layers[0], args=args, layer_name="layer1")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, args=args, logger=logger, layer_name="layer2")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, args=args, logger=logger, layer_name="layer3")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, args=args, logger=logger, layer_name="layer4")
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes, args, logger)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, args=None, logger=None, layer_name=""):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # resnet10t는 다운샘플 시 AvgPool + 1x1 Conv 사용
            if self.deep_stem and stride > 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1, args=args, logger=logger),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, args, logger),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,args,logger, block_tag=f"{layer_name}_0"))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,args=args,logger=logger, block_tag=f"{layer_name}_{i}"))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.deep_stem:
        # stem: [0]conv, [1]bn, [2]relu, [3]conv, [4]bn, [5]relu, [6]conv, [7]bn, [8]relu
            x = self.stem[0](x)
            x = self.stem[1](x)
            x = self.stem[2](x)

            # ✅ Conv2(24->32)의 입력 alpha_key 세팅(Conv2 입력은 ReLU1 출력)
            x = act_q(x, self.args, tag="stem_relu1")

            x = self.stem[3](x)
            x = self.stem[4](x)
            x = self.stem[5](x)

            # ✅ Conv3(32->64)의 입력 alpha_key 세팅(Conv3 입력은 ReLU2 출력)
            x = act_q(x, self.args, tag="stem_relu2")

            x = self.stem[6](x)
            x = self.stem[7](x)
            x = self.stem[8](x)

        else:
            # 표준 ResNet 스타일 (ResNet18 등): conv1-bn1-relu 실행
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
        # 공통 로직: Stem 이후 양자화 및 Maxpool 실행
        x = act_q(x, self.args, tag="stem_relu") 
        x = self.maxpool(x)

        # 나머지 레이어 통과
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ===== CIFAR ResNet20 (for CIFAR10) =====

class CIFARBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None, args=None, logger=None, block_tag=""):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride=stride, args=args, logger=logger)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, args=args, logger=logger)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.args = args
        self.block_tag = block_tag

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = act_q(out, self.args, tag=self.block_tag + "_relu1")

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            #print("[DBG] calling ds_in act_q:", self.block_tag + "_ds_in")
            x_ds = act_q(x, self.args, tag=self.block_tag + "_ds_in")
            identity = x_ds
            for m in self.downsample:
                identity = m(identity)

        out += identity
        out = self.relu(out)
        out = act_q(out, self.args, tag=self.block_tag + "_out")
        return out


class CIFARResNet(nn.Module):
    def __init__(self, layers, num_classes=10, args=None, logger=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.args = args
        self.inplanes = 16

        # CIFAR stem: 3x3, stride=1, no maxpool
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, args=args, logger=logger)
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, layers[0], stride=1, args=args, logger=logger, layer_name="layer1")
        self.layer2 = self._make_layer(32, layers[1], stride=2, args=args, logger=logger, layer_name="layer2")
        self.layer3 = self._make_layer(64, layers[2], stride=2, args=args, logger=logger, layer_name="layer3")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(64, num_classes, args=args, logger=logger)

    def _make_layer(self, planes, blocks, stride, args=None, logger=None, layer_name=""):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride=stride, args=args, logger=logger),
                norm_layer(planes),
            )

        layers = []
        layers.append(CIFARBasicBlock(self.inplanes, planes, stride=stride, downsample=downsample,
                                      norm_layer=norm_layer, args=args, logger=logger, block_tag=f"{layer_name}_0"))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(CIFARBasicBlock(self.inplanes, planes, stride=1, downsample=None,
                                          norm_layer=norm_layer, args=args, logger=logger, block_tag=f"{layer_name}_{i}"))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = act_q(x, self.args, tag="stem_relu")  # alpha_key 세팅 유지

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def cifar_resnet20(args=None, logger=None, num_classes=10, **kwargs):
    # ResNet20 = [3,3,3]
    return CIFARResNet([3, 3, 3], num_classes=num_classes, args=args, logger=logger)


# resnet10t 팩토리 함수 추가
def resnet10t(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], args, logger, deep_stem=True, **kwargs)
    return model



def _resnet(arch, block, layers, pretrained=None, progress=True, args=None, logger=None, **kwargs):
    model = ResNet(block, layers, args, logger, **kwargs)
    if pretrained==True:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        #model.load_state_dict(state_dict, strict=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("missing:", len(missing))
        print("unexpected:", len(unexpected))
        print("example missing:", missing[:10])
        print("example unexpected:", unexpected[:10])
    elif pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model

def resnet10(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    """
    timm의 resnet10t와 동일한 [1, 1, 1, 1] 구조를 가진 커스텀 ResNet
    """
    # 3x3 conv로 시작하는 표준 구조 (timm resnet10t와 매칭)
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], None, progress, args=args, logger=logger, **kwargs)


def resnet18(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, args=args, logger=logger,
                   **kwargs)


def resnet34(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,  args=args, logger=logger,
                   **kwargs)


def resnet101(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, args=args, logger=logger,
                   **kwargs)


def resnet152(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, args=args, logger=logger,
                   **kwargs)


def resnext50_32x4d(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], 
                   pretrained, progress, args=args, logger=logger, **kwargs)


def resnext101_32x8d(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, args=args, logger=logger, **kwargs)


def wide_resnet50_2(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, args=args, logger=logger, **kwargs)


def wide_resnet101_2(pretrained=None, progress=True, args=None, logger=None, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, args=args, logger=logger, **kwargs)


