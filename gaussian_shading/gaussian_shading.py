import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

def get_gaussian_shading_from_conf(conf, device):
    if conf.type != "gaussian_shading":
        raise ValueError("Unsupported watermark type: " + conf.type)

    if conf.get("chacha", False):
        gs = Gaussian_Shading_chacha(conf.channel_copy, conf.hw_copy, conf.fpr, conf.user_number, conf.seed, device)
    else:
        gs = Gaussian_Shading(conf.channel_copy, conf.hw_copy, conf.fpr, conf.user_number, conf.seed, device)

    return GSWatermarkAdapter(gs)

class GSWatermarkAdapter:
    def __init__(self, gs):
        self._gs = gs

    def inject_watermark(self, latents):
        # Ignore the supplied latents and use the ones from GS
        return self._gs.create_watermark_and_return_w()
    
    def check_watermark(self, latents):
        return self._gs.eval_watermark(latents)
    
    def get_state(self):
        return self._gs.get_state()

class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number, seed, device):
        self.device = device
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

        self.set_key_and_wm(seed)

    def set_key_and_wm(self, seed):
        np_g = np.random.default_rng(seed)
        self.key = np_g.bytes(32)
        self.nonce = np_g.bytes(12)
        g = torch.Generator()
        g.manual_seed(seed)
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw], generator=g).to(self.device)

    def get_state(self):
        return {
            "ch": self.ch,
            "hw": self.hw,
            "key": self.key,
            "watermark": self.watermark,
            "nonce": self.nonce,
        }

    def stream_key_encrypt(self, sd):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.to(self.device)

    def create_watermark_and_return_w(self):
        assert self.watermark is not None, "Must set up key and WM first"
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self.truncSampling(m)
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.to(self.device)

    def diffusion_inverse(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count




class Gaussian_Shading:
    def __init__(self, ch_factor, hw_factor, fpr, user_number, seed, device):
        self.device = device
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

        self.set_key_and_wm(seed)

    def set_key_and_wm(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        self.key = torch.randint(0, 2, [1, 4, 64, 64], generator=g).to(self.device)
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw], generator=g).to(self.device)

    def get_state(self):
        return {
            "ch": self.ch,
            "hw": self.hw,
            "key": self.key,
            "watermark": self.watermark,
        }

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.to(self.device)

    def create_watermark_and_return_w(self):
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self,watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_m):
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count