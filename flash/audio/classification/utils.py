import numpy as np
import pandas as pd
import librosa
import os
import requests
import zipfile
import io
import IPython
from tqdm import tqdm
from PIL import Image


def download_ESC10(data_dir):
    """Saves ESC50 files from https://github.com/karolpiczak/ESC-50 to `data_dir`"""
    # download files
    zipfile_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    r = requests.get(zipfile_url)
    zip_obj = zipfile.ZipFile(io.BytesIO(r.content))
    zip_obj.extractall(data_dir)
    # read meta data
    esc50_path = os.path.join(data_dir, 'ESC-50-master')
    df = pd.read_csv(os.path.join(esc50_path, 'meta/esc50.csv'))
    df = df[df['esc10'] == True]
    df["category"] = pd.Categorical(df["category"])
    train_df, valid_df = df[df['fold'] != 5], df[df['fold'] == 5]
    # write spectrogram images to spectrograms folder
    tqdm.pandas()
    categories = df["category"].unique()
    [os.makedirs(os.path.join(esc50_path, 'spectrograms/train', c), exist_ok=True) for c in categories]
    [os.makedirs(os.path.join(esc50_path, 'spectrograms/valid', c), exist_ok=True) for c in categories]
    train_df.progress_apply(lambda row: row_to_spec_img(
        base=esc50_path, filename=row['filename'], label=row['category'], split='train'), axis=1)
    valid_df.progress_apply(lambda row: row_to_spec_img(
        base=esc50_path, filename=row['filename'], label=row['category'], split='valid'), axis=1)


def spec_to_image(spec, eps=1e-6):
  # https://gist.github.com/hasithsura/b798c972448bed0b4fa0ab891f244d19
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    # https://gist.github.com/hasithsura/78a1fb909a4271d6287c3b273038177f
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 5 * sr:
    wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode='reflect')
    else:
    wav = wav[:5 * sr]
    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db


def row_to_spec_img(base, filename, label, split):
    output = os.path.join(base, 'spectrograms', split, label, f'${filename[:-3]}png')
    wav_path = os.path.join(base, 'audio', filename)
    spec = spec_to_image(get_melspectrogram_db(wav_path))
    img = Image.fromarray(spec)
    img.save(output)


def wav2spec(filename, visualize=False):
    spec = spec_to_image(get_melspectrogram_db(filename))
    img = Image.fromarray(spec)
    img.save(f'${filename[:-3]}png')
    if visualize:
        IPython.display.display(IPython.display.Audio(filename=filename))
        IPython.display.display(img)
