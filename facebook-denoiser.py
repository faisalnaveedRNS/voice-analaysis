from IPython import display as disp
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio


model = pretrained.dns64().cuda()
wav, sr = torchaudio.load('facebookdenoiser.wav')
wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

with torch.no_grad():
    denoised = model(wav[None])[0]
orignal=disp.Audio(wav.data.cpu().numpy(), rate=model.sample_rate)
finaldenoised=disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate)

with open('denoised.wav', 'wb') as file:
    wav_data = finaldenoised.data
    file.write(wav_data)