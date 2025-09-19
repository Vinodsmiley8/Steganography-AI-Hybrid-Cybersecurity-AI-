""" 
Steganography + AI Hybrid (Cybersecurity + AI)
Single-file Python project implementing:
 - LSB steganography embed/extract for PNG images
 - Tampering simulation (JPEG compression, noise, resize, crop)
 - Synthetic dataset generator for intact vs tampered images
 - Simple CNN (TensorFlow/Keras) to detect tampering
 - CLI utilities for embed, extract, train, detect

Dependencies:
 pip install -r requirements.txt

Usage examples (after installing deps):
 # Embed a payload into an image
 python stego_ai_hybrid.py embed --cover examples/covers/cover1.png --payload examples/payloads/secret.txt --out stego.png

 # Extract payload
 python stego_ai_hybrid.py extract --stego stego.png --out extracted.bin

 # Simulate tamper on an image
 python stego_ai_hybrid.py tamper --in examples/covers/cover1.png --out tampered.jpg --method jpeg

 # Create dataset (small)
 python stego_ai_hybrid.py createdata --covers-dir examples/covers --payloads-dir examples/payloads --outdir data/ --num 200

 # Train detector
 python stego_ai_hybrid.py train --datadir data/ --model out_model.h5 --epochs 10

 # Predict tampering
 python stego_ai_hybrid.py detect --model out_model.h5 --image stego.png

Note: This is a starting point and meant for research/learning. For production-grade stego or forensic tools, more robust techniques and datasets are required.
"""

import argparse
import os
import io
import random
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import math

# Optional import for training
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------------------- LSB Steganography ----------------------

def _to_bitarray(data_bytes):
    bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
    return bits

def _from_bitarray(bits):
    arr = np.packbits(bits.astype(np.uint8))
    return arr.tobytes()

def embed_lsb(cover_path, payload_path, out_path, bits_per_channel=1):
    """Embed payload file bytes into cover PNG using LSB on RGB channels.
    bits_per_channel: how many LSBs to use per color channel (1-4 recommended)
    """
    assert 1 <= bits_per_channel <= 4
    img = Image.open(cover_path).convert('RGBA')
    w, h = img.size
    pixels = np.array(img)
    channels = 3  # use RGB, ignore alpha for embedding

    with open(payload_path, 'rb') as f:
        payload = f.read()
    # Prepend payload length (4 bytes) so extraction knows size
    plen = len(payload)
    header = plen.to_bytes(4, byteorder='big')
    data = header + payload

    bits = _to_bitarray(data)
    capacity = w * h * channels * bits_per_channel
    if bits.size > capacity:
        raise ValueError(f"Payload too large for cover image. Need {bits.size} bits, capacity {capacity} bits.")

    # pad bits
    pad_len = capacity - bits.size
    if pad_len > 0:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])

    # write bits into LSBs
    flat = pixels[..., :3].flatten()  # RGB flattened
    # precompute clear mask (safe uint8 mask)
    lsb_mask = (1 << bits_per_channel) - 1       # e.g., 1->0b1, 2->0b11
    clear_mask = 0xFF ^ lsb_mask                 # e.g., if lsb_mask=0b11, clear_mask=0b11111100

    n_pixels = flat.shape[0]
    n_values = bits.size // bits_per_channel     # number of pixel-chunks to write
    if n_values > n_pixels:
        raise ValueError("Internal error: computed more write values than available pixels.")

    for i in range(n_values):
        byte_idx = i * bits_per_channel
        chunk = bits[byte_idx: byte_idx + bits_per_channel]
        val = 0
        for b in chunk:
            val = (val << 1) | int(b)
        # ensure val fits in bits_per_channel
        val = int(val) & lsb_mask
        # apply mask and OR the value (all operands are in 0..255 range)
        flat[i] = (int(flat[i]) & clear_mask) | val

    # reshape and save
    new_rgb = flat.reshape((h, w, 3))
    new_pixels = np.concatenate([new_rgb, pixels[..., 3:4]], axis=2)
    out_img = Image.fromarray(new_pixels.astype(np.uint8), 'RGBA')
    out_img.save(out_path, optimize=True)
    print(f"Embedded payload ({plen} bytes) into {out_path}")


def extract_lsb(stego_path, out_payload_path, bits_per_channel=1):
    img = Image.open(stego_path).convert('RGBA')
    w, h = img.size
    pixels = np.array(img)
    channels = 3
    capacity = w * h * channels * bits_per_channel

    flat = pixels[..., :3].flatten()
    bits = np.zeros(capacity, dtype=np.uint8)
    for i in range(capacity // bits_per_channel):
        v = flat[i] & ((1 << bits_per_channel) - 1)
        # convert v to bits_per_channel bits
        for j in range(bits_per_channel-1, -1, -1):
            bits[i*bits_per_channel + (bits_per_channel-1-j)] = (v >> j) & 1

    # first 32 bits = payload length
    header_bits = bits[:32]
    header_bytes = _from_bitarray(header_bits)
    plen = int.from_bytes(header_bytes, byteorder='big')
    if plen <= 0 or plen > (capacity // 8 - 4):
        print("Warning: suspicious payload length extracted. It may be tampered or wrong bits_per_channel.")

    total_bits_needed = (4 + plen) * 8
    data_bits = bits[:total_bits_needed]
    data = _from_bitarray(data_bits)
    payload = data[4:4+plen]
    with open(out_payload_path, 'wb') as f:
        f.write(payload)
    print(f"Extracted payload to {out_payload_path} ({plen} bytes)")

# ---------------------- Tampering / Augmentations ----------------------

def tamper_jpeg(image: Image.Image, quality=30):
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGBA')

def tamper_noise(image: Image.Image, std=10):
    arr = np.array(image).astype(np.int16)
    noise = np.random.normal(0, std, arr[..., :3].shape).astype(np.int16)
    arr[..., :3] += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.concatenate([arr[..., :3], np.array(image)[..., 3:4]], axis=2))

def tamper_resize(image: Image.Image, scale=0.9):
    w,h = image.size
    neww = max(1, int(w*scale))
    newh = max(1, int(h*scale))
    img2 = image.resize((neww,newh), Image.BICUBIC)
    return img2.resize((w,h), Image.BICUBIC)

def tamper_crop(image: Image.Image, crop_frac=0.1):
    w,h = image.size
    cw = int(w * (1-crop_frac))
    ch = int(h * (1-crop_frac))
    left = random.randint(0, w - cw)
    top = random.randint(0, h - ch)
    cropped = image.crop((left, top, left+cw, top+ch))
    return cropped.resize((w,h), Image.BICUBIC)

def tamper_methods():
    return ['jpeg','noise','resize','crop']

def apply_tamper(image: Image.Image, method=None):
    if method is None:
        method = random.choice(tamper_methods())
    if method == 'jpeg':
        return tamper_jpeg(image, quality=random.randint(15,60))
    if method == 'noise':
        return tamper_noise(image, std=random.uniform(5,25))
    if method == 'resize':
        return tamper_resize(image, scale=random.uniform(0.7,0.98))
    if method == 'crop':
        return tamper_crop(image, crop_frac=random.uniform(0.02,0.2))
    return image

# ---------------------- Dataset Generation ----------------------

def create_dataset(covers_dir, payloads_dir, out_dir, num=500, bits_per_channel=1, tamper_fraction=0.5):
    os.makedirs(out_dir, exist_ok=True)
    covers = [os.path.join(covers_dir,f) for f in os.listdir(covers_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    payloads = [os.path.join(payloads_dir,f) for f in os.listdir(payloads_dir)]
    if not covers:
        raise ValueError('No cover images found')
    if not payloads:
        raise ValueError('No payload files found')

    for i in tqdm(range(num)):
        cover = Image.open(random.choice(covers)).convert('RGBA')
        cover_path = os.path.join(out_dir, f'cover_{i}.png')
        cover.save(cover_path)
        payload = random.choice(payloads)
        stego_path = os.path.join(out_dir, f'stego_{i}.png')
        # embed payload
        tmp_payload = payload
        # If payload too big for the cover, skip
        try:
            embed_lsb(cover_path, tmp_payload, stego_path, bits_per_channel=bits_per_channel)
        except Exception as e:
            # skip cases where payload too big
            continue
        # decide if tampered
        if random.random() < tamper_fraction:
            tam_img = apply_tamper(Image.open(stego_path).convert('RGBA'))
            tam_path = os.path.join(out_dir, f'tampered_{i}.png')
            tam_img.save(tam_path)
            label = 'tampered'
            # move tampered to images dir
            os.replace(tam_path, os.path.join(out_dir, f'image_{i}.png'))
            # store label
            with open(os.path.join(out_dir, f'label_{i}.txt'), 'w') as f:
                f.write('1')
        else:
            os.replace(stego_path, os.path.join(out_dir, f'image_{i}.png'))
            with open(os.path.join(out_dir, f'label_{i}.txt'), 'w') as f:
                f.write('0')
    print('Dataset generation finished in', out_dir)

# ---------------------- Model / Training ----------------------

def build_simple_cnn(input_shape=(128,128,4)):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available')
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_image_for_model(path, size=(128,128)):
    img = Image.open(path).convert('RGBA')
    img = img.resize(size, Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def data_generator_from_dir(datadir, batch_size=16, size=(128,128)):
    files = [f for f in os.listdir(datadir) if f.startswith('image_') and f.endswith('.png')]
    idx = 0
    while True:
        batch_files = files[idx: idx+batch_size]
        if not batch_files:
            idx = 0
            random.shuffle(files)
            continue
        X = []
        y = []
        for f in batch_files:
            base = f.split('.')[0].split('_')[1]
            imgpath = os.path.join(datadir, f)
            labelpath = os.path.join(datadir, f'label_{base}.txt')
            if not os.path.exists(labelpath):
                continue
            lab = int(open(labelpath).read().strip())
            arr = load_image_for_model(imgpath, size=size)
            X.append(arr)
            y.append(lab)
        idx += batch_size
        if X:
            yield np.array(X), np.array(y).astype(np.float32)

def train_detector(datadir, model_out, epochs=10, batch_size=16, size=(128,128)):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available for training')
    model = build_simple_cnn(input_shape=(size[0], size[1], 4))
    gen = data_generator_from_dir(datadir, batch_size=batch_size, size=size)
    steps = max(10,  len([f for f in os.listdir(datadir) if f.startswith('image_')]) // batch_size)
    model.fit(gen, steps_per_epoch=steps, epochs=epochs)
    model.save(model_out)
    print('Saved model to', model_out)

def predict_tamper(model_path, image_path, size=(128,128)):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow not available for inference')
    model = models.load_model(model_path)
    arr = load_image_for_model(image_path, size=size)
    pred = model.predict(np.expand_dims(arr, axis=0))[0,0]
    print(f'Predicted tamper probability: {pred:.4f} (closer to 1 = tampered)')
    return pred

# ---------------------- CLI ----------------------

def main():
    p = argparse.ArgumentParser(description='Steganography + AI Hybrid')
    sub = p.add_subparsers(dest='cmd')

    e = sub.add_parser('embed')
    e.add_argument('--cover', required=True)
    e.add_argument('--payload', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--bits', type=int, default=1)

    ex = sub.add_parser('extract')
    ex.add_argument('--stego', required=True)
    ex.add_argument('--out', required=True)
    ex.add_argument('--bits', type=int, default=1)

    t = sub.add_parser('tamper')
    t.add_argument('--in', dest='infile', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--method', choices=tamper_methods(), default='jpeg')

    cd = sub.add_parser('createdata')
    cd.add_argument('--covers-dir', required=True)
    cd.add_argument('--payloads-dir', required=True)
    cd.add_argument('--outdir', required=True)
    cd.add_argument('--num', type=int, default=200)
    cd.add_argument('--bits', type=int, default=1)

    tr = sub.add_parser('train')
    tr.add_argument('--datadir', required=True)
    tr.add_argument('--model', required=True)
    tr.add_argument('--epochs', type=int, default=5)

    det = sub.add_parser('detect')
    det.add_argument('--model', required=True)
    det.add_argument('--image', required=True)

    args = p.parse_args()
    if args.cmd == 'embed':
        embed_lsb(args.cover, args.payload, args.out, bits_per_channel=args.bits)
    elif args.cmd == 'extract':
        extract_lsb(args.stego, args.out, bits_per_channel=args.bits)
    elif args.cmd == 'tamper':
        img = Image.open(args.infile).convert('RGBA')
        img2 = apply_tamper(img, method=args.method)
        img2.save(args.out)
        print('Saved tampered image to', args.out)
    elif args.cmd == 'createdata':
        create_dataset(args.covers_dir, args.payloads_dir, args.outdir, num=args.num, bits_per_channel=args.bits)
    elif args.cmd == 'train':
        train_detector(args.datadir, args.model, epochs=args.epochs)
    elif args.cmd == 'detect':
        predict_tamper(args.model, args.image)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
