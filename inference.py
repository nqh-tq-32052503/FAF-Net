import os
import random
import numpy as np
import torch
import copy
import torchaudio
from tqdm import tqdm
from utils import prepare_references, istft, stft, batch_match
from model import ComplexUnet, ComplexUnet_


class FAFExecutive(object):
    def __init__(self, list_references, num_stages=1):
        self.reference = prepare_references(list_references)
        self.num_stages = num_stages
        self.init_model()
    
    def init_model(self):
        m = ComplexUnet()
        m.load_state_dict(torch.load("./ckpt_1.pth", map_location="cpu"))
        m.eval().cuda()
        self.models = [m]
        if self.num_stages == 2:
            m_ = ComplexUnet_()
            m_.load_state_dict(torch.load("./ckpt_2.pth", map_location="cpu"))
            m_.eval().cuda()
            self.models.append(m_)
    
    def inference(self, audio_path):
        noisy_audio, _ = torchaudio.load(audio_path)
        noisy = noisy_audio.cuda()
        ref_mfcc = self.reference["mfccs"].unsqueeze(0).cuda()
        ref_stft = self.reference["stfts"].unsqueeze(0).cuda()
        noisy_stft = stft(noisy)
        length = noisy.shape[1]
        idx = batch_match(noisy, ref_mfcc)
        with torch.no_grad():
            pre_spectra = self.models[0](noisy_stft, ref_stft, idx)
        enhanced_wav = istft(pre_spectra, length)
        
        if self.num_stages == 2:
            idx = batch_match(enhanced_wav, ref_mfcc)
            with torch.no_grad():
                pre_spectra = self.models[1](noisy_stft, pre_spectra, ref_stft, idx)
            enhanced_wav = istft(pre_spectra, length)
        
        output = enhanced_wav.detach().squeeze().cpu().numpy()
        return output
        
