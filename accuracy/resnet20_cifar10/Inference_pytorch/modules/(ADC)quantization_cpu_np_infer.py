import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer,wage_quantizer
import numpy as np


_ADC_RANGE = {}          # name -> (minQ, maxQ)
_ADC_CALIB_ENABLED = False

def adc_calib_enable(flag: bool = True):
    global _ADC_CALIB_ENABLED
    _ADC_CALIB_ENABLED = bool(flag)

def adc_calib_disable():
    adc_calib_enable(False)

def adc_calib_reset():
    _ADC_RANGE.clear()

def adc_calib_is_enabled():
    return _ADC_CALIB_ENABLED

def adc_range_update(name, x):
    # running absmax update (stable + cheap)
    with torch.no_grad():
        a = float(x.detach().abs().max().item())
        if not (a > 0.0) or (a != a):
            return
        prev = _ADC_RANGE.get(name, None)
        if prev is None:
            _ADC_RANGE[name] = (-a, a)
        else:
            # expand range if needed
            prev_min, prev_max = prev
            new_min = min(prev_min, -a)
            new_max = max(prev_max, +a)
            _ADC_RANGE[name] = (new_min, new_max)

def adc_range_get(name, default=(-1.0, 1.0)):
    return _ADC_RANGE.get(name, default)


def LinearQuantizeOut_fixedrange(x, bit, minQ_fixed, maxQ_fixed):
    """
    고정 범위(minQ_fixed,maxQ_fixed)로 clamp 후 uniform quant.
    NOTE: 지금은 너 원래 구현을 따라 step=delta*2^-bit 로 둠.
          (정확한 2^bit 레벨이라면 delta/(2^bit-1)도 옵션)
    """
    delta = float(maxQ_fixed - minQ_fixed)
    if not (delta > 0.0):
        return x

    stepSize = delta * (2.0 ** (-bit))
    if not (stepSize > 0.0):
        return x

    y = x.clamp(minQ_fixed, maxQ_fixed)
    index = torch.clamp(torch.floor((y - minQ_fixed) / stepSize), 0, (2.0 ** bit - 1.0))
    yq = index * stepSize + minQ_fixed
    return yq

def _use_bitserial_hw_path(model):
    if model is None:
        return False
    m = str(model).lower()
    return m.startswith('vgg') or ('resnet' in m)


# --- calibration store (module-global) ---
_ACT_ALPHA = {}
_NEXT_ACT_ALPHA_KEY = None

def calib_set_alpha_dict(d: dict):
    global _ACT_ALPHA
    _ACT_ALPHA = {str(k): float(v) for k, v in (d or {}).items()}


def set_next_alpha_key(key: str | None):
    global _NEXT_ACT_ALPHA_KEY
    _NEXT_ACT_ALPHA_KEY = None if key is None else str(key)

# def peek_next_alpha_key(default=None):
#     # consume(비우기) 하지 않는다
#     return _NEXT_ACT_ALPHA_KEY if _NEXT_ACT_ALPHA_KEY is not None else default

def pop_next_alpha_key(default=None):
    global _NEXT_ACT_ALPHA_KEY
    v = _NEXT_ACT_ALPHA_KEY if _NEXT_ACT_ALPHA_KEY is not None else default
    _NEXT_ACT_ALPHA_KEY = None
    return v

def calib_get_alpha(key: str, default=None):
    return _ACT_ALPHA.get(str(key), default)

def _act_to_uint_code_alpha_aware(x, bitsA, alpha=None, alpha_key=None, eps=1e-12):
    L = (2 ** bitsA) - 1

    # 1) alpha priority: explicit alpha > alpha_key(lookup) > runtime max fallback
    if alpha is None and alpha_key is not None:
        alpha = calib_get_alpha(alpha_key, None)

    with torch.no_grad():
        if alpha is None:
            alpha = float(x.detach().max().clamp_min(eps).item())
        alpha = float(max(alpha, eps))

    x_clip = x.clamp(0.0, alpha)
    q = torch.round(L * x_clip / alpha)  # 0..L
    return q, alpha

def _act_quant_dequant_alpha_key_ste(x, bitsA, alpha_key=None, eps=1e-6, ste=True):
    """
    unsigned ReLU activation quant/dequant using alpha_key:
      x -> clamp[0,alpha] -> round(levels*x/alpha) -> dequant -> float
    ste=True면 gradient는 x로 통과시키는 STE 적용.
    """
    if bitsA == -1 or bitsA >= 16:
        return x, None

    alpha = None
    if alpha_key is not None:
        alpha = calib_get_alpha(alpha_key, None)

    # fallback: runtime max
    with torch.no_grad():
        if alpha is None:
            alpha = float(x.detach().max().clamp_min(eps).item())
        alpha = float(alpha)
        if (not (alpha > 0.0)) or (alpha != alpha) or (alpha < eps):
            alpha = eps

    levels = (1 << bitsA) - 1

    # tensorize alpha on same device/dtype to avoid float inf issues
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)

    # unsigned assumption: ReLU domain
    x_u = x.clamp_min(0.0)
    x_c = x_u.clamp(0.0, alpha_t)

    q = torch.round(x_c * (levels / alpha_t))
    y = q * (alpha_t / levels)

    # numeric safety
    y = torch.nan_to_num(y, nan=0.0, posinf=alpha, neginf=0.0)

    if ste:
        y = x_u + (y - x_u).detach()

    return y, alpha





class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
                 wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,vari=0,t=0,v=0,detect=0,target=0,debug = 0, name = 'Qconv', model = None,
                 parallelRead=128):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.parallelRead = parallelRead
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)
        

    def forward(self, input):
        
        # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        # weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        # outputOrignal= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # bitWeight = int(self.wl_weight)
        # bitActivation = int(self.wl_input)

        w_scaled = self.weight * self.scale                          # (A) scale domain
        wq_scaled = w_scaled + (wage_quantizer.Q(w_scaled, self.wl_weight) - w_scaled).detach()
    
        # retention을 scale-domain에서 하고 싶으면 여기에 두는 게 일관적
        wq_scaled = wage_quantizer.Retention(wq_scaled, self.t, self.v, self.detect, self.target)
    
        weight = wq_scaled / self.scale                              # (B) restored to [-1,1] domain
    
        # ------------------------------------------------------------
        # 2) use "weight" everywhere (NO output /= self.scale at end)
        # ------------------------------------------------------------
        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

    
        outputOrignal = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        xdec_gain = 16.0 if bitWeight == 4 else 1.0

        if self.inference == 1 and _use_bitserial_hw_path(self.model):
            alpha_key = pop_next_alpha_key(default=None)
            #print("[DBG]", self.name, "alpha_key =", alpha_key)

            # Parallel read is only supported for single-level cells

            if input.shape[1] != 3 and alpha_key is None:
                print("[WARN]", self.name, "alpha_key is None (tag mismatch or act_q missing)")

            if self.parallelRead < self.subArray:
                self.subArray = self.parallelRead

            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1/onoffratio
        
            output = torch.zeros_like(outputOrignal)
            del outputOrignal
            cellRange = 2**self.cellBit   # cell precision is 4
        
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:,:,:,:] = (cellRange-1)*(upper+lower)/2
            numSubArray = int(weight.shape[1]/self.subArray)

            if weight.shape[1] == 3:
                print(self.name, "in_minmax", float(input.min()), float(input.max()))
                
            else :    
                inputQ, alpha_in = _act_to_uint_code_alpha_aware(input, bitActivation, alpha_key=alpha_key)
                sat = (input > alpha_in).float().mean().item()
                print(self.name, "alpha_in", alpha_in, "sat_ratio", sat, "in_minmax", float(input.min()), float(input.max()))

            for i in range (self.weight.shape[2]):
                for j in range (self.weight.shape[3]):
                    # need to divide to different subArray
                    # cut into different subArrays
                    if numSubArray == 0:
                        mask = torch.zeros_like(weight)
                        mask[:,:,i,j] = 1
                        if weight.shape[1] == 3:
                        #if False :
                            # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                            outputP = torch.zeros_like(output)
                            outputD = torch.zeros_like(output)
                            for k in range (int(bitWeight/self.cellBit)):
                                remainder = torch.fmod(X_decimal, cellRange)*mask
                                # retention
                                remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                # Now also consider weight has on/off ratio effects
                                # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                remainderQ = remainderQ * xdec_gain
                                dummyP_eff = dummyP * xdec_gain
                            
                                outputPartial= F.conv2d(input, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputDummyPartial= F.conv2d(input, dummyP_eff*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                scaler = cellRange**k
                                outputP = outputP + outputPartial*scaler*2/(1-1/onoffratio)
                                outputD = outputD + outputDummyPartial*scaler*2/(1-1/onoffratio)
                            outputP = outputP - outputD
                            output = output + outputP
                        else:
                            inputQ, alpha_in = _act_to_uint_code_alpha_aware(input, bitActivation, alpha_key=alpha_key)
                            outputIN = torch.zeros_like(output)
                            for z in range(bitActivation):
                                inputB = torch.fmod(inputQ, 2)
                                inputQ = torch.round((inputQ-inputB)/2)
                                outputP = torch.zeros_like(output)
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                weight = weight.clamp(-1.0, 1.0)
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                    remainderQ = remainderQ * xdec_gain
                                    dummyP_eff = dummyP * xdec_gain
                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP_eff*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    # --- ADC quant (fixed-range) + optional calibration update ---
                                    if adc_calib_is_enabled():
                                        adc_range_update(self.name, outputPartial)
                                        adc_range_update(self.name, outputDummyPartial)
                                        outputPartialQ = outputPartial
                                        outputDummyPartialQ = outputDummyPartial
                                    else:
                                        minQ_fixed, maxQ_fixed = adc_range_get(self.name, default=(-1.0, 1.0))
                                        outputPartialQ = LinearQuantizeOut_fixedrange(outputPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)
                                        outputDummyPartialQ = LinearQuantizeOut_fixedrange(outputDummyPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)

                                                                        
                                    # outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    # outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                    scaler = cellRange**k
                                    outputP = outputP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                scalerIN = 2**z
                                outputIN = outputIN + (outputP - outputD)*scalerIN
                            #output = output + outputIN/(2**bitActivation)
                            output = output + outputIN * (alpha_in / (2 ** bitActivation - 1))

                    else:
                        inputQ, alpha_in = _act_to_uint_code_alpha_aware(input, bitActivation, alpha_key=alpha_key)
                        outputIN = torch.zeros_like(output)
                        for z in range(bitActivation):
                            inputB = torch.fmod(inputQ, 2)
                            inputQ = torch.round((inputQ-inputB)/2)
                            outputP = torch.zeros_like(output)
                            for s in range(numSubArray):
                                mask = torch.zeros_like(weight)
                                mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                                # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                                weight = weight.clamp(-1.0, 1.0)
                                X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                                outputSP = torch.zeros_like(output)
                                outputD = torch.zeros_like(output)
                                for k in range (int(bitWeight/self.cellBit)):
                                    remainder = torch.fmod(X_decimal, cellRange)*mask
                                    # retention
                                    remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                                    X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    # Now also consider weight has on/off ratio effects
                                    # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                                    # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                                    remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                                    remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                                    remainderQ = remainderQ * xdec_gain
                                    dummyP_eff = dummyP * xdec_gain

                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP_eff*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    # Add ADC quanization effects here !!!
                                    if adc_calib_is_enabled():
                                        adc_range_update(self.name, outputPartial)
                                        adc_range_update(self.name, outputDummyPartial)
                                        outputPartialQ = outputPartial
                                        outputDummyPartialQ = outputDummyPartial
                                    else:
                                        minQ_fixed, maxQ_fixed = adc_range_get(self.name, default=(-1.0, 1.0))
                                        outputPartialQ = LinearQuantizeOut_fixedrange(outputPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)
                                        outputDummyPartialQ = LinearQuantizeOut_fixedrange(outputDummyPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)

                              

                                    # outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                                    # outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                                    scaler = cellRange**k
                                    outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                                    outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                                # !!! Important !!! the dummy need to be multiplied by a ratio
                                outputSP = outputSP - outputD  # minus dummy column
                                outputP = outputP + outputSP
                            scalerIN = 2**z
                            outputIN = outputIN + outputP*scalerIN
                        output = output + outputIN * (alpha_in / (2 ** bitActivation - 1))

            output = output/(2**bitWeight)   # since weight range was convert from [-1, 1] to [-256, 256]
            if bitWeight == 4:
                output = output / xdec_gain
            #     gamma = 15/16
            #     output *= gamma 
            #output = output * (2.0 / (2**bitWeight - 1))
            #print(self.name, "out min/max", float(output.min()), float(output.max()), "mean", float(output.mean()))


            
        elif self.inference == 1:
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            # original WAGE QCov2d
            # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            # weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            # weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            # output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            w_scaled = self.weight * self.scale            # forward에서도 scaling
            wq_scaled = w_scaled + (wage_quantizer.Q(w_scaled, self.wl_weight) - w_scaled).detach()
            weight = wq_scaled / self.scale
            weight = weight.clamp(-1.0, 1.0)  

            alpha_key = pop_next_alpha_key(default=None)

            if weight.shape[1] == 3:
                input_q = input
            else :
                
                if alpha_key is not None:
                    input_q, _ = _act_quant_dequant_alpha_key_ste(
                        input, int(self.wl_input), alpha_key=alpha_key, eps=1e-6, ste=True
                    )
                    alpha = calib_get_alpha(alpha_key, None) if alpha_key else None
                    #print(self.name, "alpha_key", alpha_key, "alpha", alpha, "in_ch", input.shape[1])
                else:
                    #print("[WARN]", self.name, "alpha_key is None (tag mismatch or act_q missing)")
                    input_q = input
            

            output = F.conv2d(input_q, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            #print(self.name, "out min/max", float(output.min()), float(output.max()), "mean", float(output.mean()))


            
        #output = output/self.scale
        #output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)
        
        return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,
	             wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,vari=0,t=0,v=0,detect=0,target=0,debug = 0, name ='Qlinear', model = None,
                 parallelRead=128):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_input = wl_input
        self.wl_error = wl_error
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.parallelRead = parallelRead
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.name = name
        self.model = model
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

    def forward(self, input):

        # weight1 = self.weight * self.scale # + (self.weight - self.weight * self.scale).detach()
        # weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        # outputOrignal = F.linear(input, weight, self.bias)
        # output = torch.zeros_like(outputOrignal)

        bitWeight = int(self.wl_weight)
        bitActivation = int(self.wl_input)

        w_scaled = self.weight * self.scale                          # (A) scale domain
        wq_scaled = w_scaled + (wage_quantizer.Q(w_scaled, self.wl_weight) - w_scaled).detach()
    
        # retention을 scale-domain에서 하고 싶으면 여기에 두는 게 일관적
        wq_scaled = wage_quantizer.Retention(wq_scaled, self.t, self.v, self.detect, self.target)
    
        weight = wq_scaled / self.scale    

        outputOrignal = F.linear(input, weight, self.bias)
        output = torch.zeros_like(outputOrignal)
        xdec_gain = 16.0 if bitWeight == 4 else 1.0
        

        if self.inference == 1 and _use_bitserial_hw_path(self.model):
            # Parallel read is only supported for single-level cells
            if self.parallelRead < self.subArray:
                self.subArray = self.parallelRead

            # set parameters for Hardware Inference
            onoffratio = self.onoffratio
            upper = 1
            lower = 1/onoffratio
            output = torch.zeros_like(outputOrignal)
            cellRange = 2**self.cellBit   # cell precision is 4
            # Now consider on/off ratio
            dummyP = torch.zeros_like(weight)
            dummyP[:,:] = (cellRange-1)*(upper+lower)/2
            # need to divide to different subArray
            numSubArray = int(weight.shape[1]/self.subArray)
            alpha_key = pop_next_alpha_key(default=None)

            if input.shape[1] != 3 and alpha_key is None:
                print("[WARN]", self.name, "alpha_key is None (tag mismatch or act_q missing)")

            
                        

            if numSubArray == 0:
                mask = torch.zeros_like(weight)
                mask[:,:] = 1
                # quantize input into binary sequence
                inputQ, alpha_in = _act_to_uint_code_alpha_aware(input, bitActivation, alpha_key=alpha_key)
                sat = (input > alpha_in).float().mean().item()
                print(self.name, "alpha_in", alpha_in, "sat_ratio", sat, "in_minmax", float(input.min()), float(input.max()))

                outputIN = torch.zeros_like(outputOrignal)

                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                    weight = weight.clamp(-1.0, 1.0)
                    X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                    outputP = torch.zeros_like(outputOrignal)
                    outputD = torch.zeros_like(outputOrignal)
                    for k in range (int(bitWeight/self.cellBit)):
                        remainder = torch.fmod(X_decimal, cellRange)*mask
                        # retention
                        remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                        X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                        # Now also consider weight has on/off ratio effects
                        # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                        # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]
                        remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                        remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                        remainderQ = remainderQ * xdec_gain
                        dummyP_eff = dummyP * xdec_gain
                        
                        outputPartial= F.linear(inputB, remainderQ*mask, self.bias)
                        outputDummyPartial= F.linear(inputB, dummyP_eff*mask, self.bias)
                        # Add ADC quanization effects here !!!
                        if adc_calib_is_enabled():
                            adc_range_update(self.name, outputPartial)
                            adc_range_update(self.name, outputDummyPartial)
                            outputPartialQ = outputPartial
                            outputDummyPartialQ = outputDummyPartial
                        else:
                            minQ_fixed, maxQ_fixed = adc_range_get(self.name, default=(-1.0, 1.0))
                            outputPartialQ = LinearQuantizeOut_fixedrange(outputPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)
                            outputDummyPartialQ = LinearQuantizeOut_fixedrange(outputDummyPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)

                    

                        # outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                        # outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                        scaler = cellRange**k
                        outputP = outputP + outputPartialQ*scaler*2/(1-1/onoffratio)
                        outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                    scalerIN = 2**z
                    outputIN = outputIN + (outputP - outputD)*scalerIN
                output = output + outputIN * (alpha_in / (2 ** bitActivation - 1))
            else:
                inputQ, alpha_in = _act_to_uint_code_alpha_aware(input, bitActivation, alpha_key=alpha_key)
                sat = (input > alpha_in).float().mean().item()
                print(self.name, "alpha_in", alpha_in, "sat_ratio", sat, "in_minmax", float(input.min()), float(input.max()))
                outputIN = torch.zeros_like(outputOrignal)
                for z in range(bitActivation):
                    inputB = torch.fmod(inputQ, 2)
                    inputQ = torch.round((inputQ-inputB)/2)
                    outputP = torch.zeros_like(outputOrignal)
                    for s in range(numSubArray):
                        mask = torch.zeros_like(weight)
                        mask[:,(s*self.subArray):(s+1)*self.subArray] = 1
                        # after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
                        weight = weight.clamp(-1.0, 1.0)
                        X_decimal = torch.round((2**bitWeight - 1)/2 * (weight+1) + 0)*mask
                        outputSP = torch.zeros_like(outputOrignal)
                        outputD = torch.zeros_like(outputOrignal)
                        for k in range (int(bitWeight/self.cellBit)):
                            remainder = torch.fmod(X_decimal, cellRange)*mask
                            # retention
                            remainder = wage_quantizer.Retention(remainder,self.t,self.v,self.detect,self.target)
                            X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                            # Now also consider weight has on/off ratio effects
                            # Here remainder is the weight mapped to Hardware, so we introduce on/off ratio in this value
                            # the range of remainder is [0, cellRange-1], we truncate it to [lower, upper]*(cellRange-1)
                            remainderQ = (upper-lower)*(remainder-0)+(cellRange-1)*lower   # weight cannot map to 0, but to Gmin
                            remainderQ = remainderQ + remainderQ*torch.normal(0., torch.full(remainderQ.size(),self.vari, device='cuda'))
                            remainderQ = remainderQ * xdec_gain
                            dummyP_eff = dummyP * xdec_gain
                            
                            outputPartial= F.linear(inputB, remainderQ*mask, self.bias)
                            outputDummyPartial= F.linear(inputB, dummyP_eff*mask, self.bias)
                            # Add ADC quanization effects here !!!
                            if adc_calib_is_enabled():
                                adc_range_update(self.name, outputPartial)
                                adc_range_update(self.name, outputDummyPartial)
                                outputPartialQ = outputPartial
                                outputDummyPartialQ = outputDummyPartial
                            else:
                                minQ_fixed, maxQ_fixed = adc_range_get(self.name, default=(-1.0, 1.0))
                                outputPartialQ = LinearQuantizeOut_fixedrange(outputPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)
                                outputDummyPartialQ = LinearQuantizeOut_fixedrange(outputDummyPartial, self.ADCprecision, minQ_fixed, maxQ_fixed)

                        
                            
                            # outputPartialQ = wage_quantizer.LinearQuantizeOut(outputPartial, self.ADCprecision)
                            # outputDummyPartialQ = wage_quantizer.LinearQuantizeOut(outputDummyPartial, self.ADCprecision)
                            scaler = cellRange**k
                            outputSP = outputSP + outputPartialQ*scaler*2/(1-1/onoffratio)
                            outputD = outputD + outputDummyPartialQ*scaler*2/(1-1/onoffratio)
                        outputSP = outputSP - outputD  # minus dummy column
                        outputP = outputP + outputSP
                    scalerIN = 2**z
                    outputIN = outputIN + outputP*scalerIN
                output = output + outputIN * (alpha_in / (2 ** bitActivation - 1))
            output = output/(2**bitWeight)
            if bitWeight == 4:
                output = output / xdec_gain
            #     gamma = 15/16
            #     output *= gamma 
            #output = output * (2.0 / (2**bitWeight - 1))
            #print(self.name, "out min/max", float(output.min()), float(output.max()), "mean", float(output.mean()))

        
        elif self.inference == 1:
            # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            # weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            # weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            # input = wage_quantizer.Q(input,self.wl_input)
            # output= F.linear(input, weight, self.bias)
            # output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)

            w_scaled = self.weight * self.scale            # forward에서도 scaling
            wq_scaled = w_scaled + (wage_quantizer.Q(w_scaled, self.wl_weight) - w_scaled).detach()
            weight = wq_scaled / self.scale 
            input = wage_quantizer.Q(input,self.wl_input)
            output = F.linear(input, weight, self.bias)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
            

        else:
            # original WAGE QCov2d
            # weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            # weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            # weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            # output = F.linear(input, weight, self.bias)


       
            w_scaled = self.weight * self.scale
            wq_scaled = w_scaled + (wage_quantizer.Q(w_scaled, self.wl_weight) - w_scaled).detach()
            weight = wq_scaled / self.scale
            weight = weight.clamp(-1.0, 1.0)

            alpha_key = pop_next_alpha_key(default=None)
            if alpha_key is not None:
                input_q, _ = _act_quant_dequant_alpha_key_ste(
                    input, int(self.wl_input), alpha_key=alpha_key, eps=1e-6, ste=True
                )
                alpha = calib_get_alpha(alpha_key, None) if alpha_key else None
                #print(self.name, "alpha_key", alpha_key, "alpha", alpha, "in_ch", input.shape[1])
                
            else:
                input_q = input


            


            output = F.linear(input_q, weight, self.bias)
            #print(self.name, "out min/max", float(output.min()), float(output.max()), "mean", float(output.mean()))

            
        #output = output/self.scale
        
        # x = output.detach()
        # clip_ratio = ((x > 1.0) | (x < -1.0)).float().mean().item()
        # print(self.name, "pre-clip ratio", clip_ratio, "min/max", x.min().item(), x.max().item())

        #output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)
        
        return output

