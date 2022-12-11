import torch
from train import load_checkpoint
from glow import WaveGlow
from scipy.io.wavfile import write
import numpy as np

from tacotron2.model import Tacotron2
from tacotron2.hparams import create_hparams
from tacotron2.train import load_model as load_tacotron
from tacotron2.text import text_to_sequence

SAMPLING_RATE = 22050
WAVEGLOW_MODEL = "models/mbailey_wg.pt"
TACOTRON_MODEL = "models/tacotron2.pt"

# Create our inference Tacotron model
hparams = create_hparams()
hparams.sampling_rate = SAMPLING_RATE

tacotron = load_tacotron(hparams)
tacotron.load_state_dict(torch.load(TACOTRON_MODEL)['state_dict'])
_ = tacotron.cuda().eval().half()


waveglow = torch.load(WAVEGLOW_MODEL)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
# denoiser = Denoiser(waveglow)

text = "Waveglow is really awesome!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)
).cuda().long()


mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(sequence)
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
audio_numpy = audio[0].data.cpu().numpy()
write("audio.wav", SAMPLING_RATE, audio_numpy)