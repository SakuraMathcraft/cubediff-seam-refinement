"""Measure geometric overlap RGB L1 from six untrimmed PNG faces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .geometry import FACE_NAMES, overlap_band_diagnostic


def measure(directory):
    faces, fingerprints = [], {}
    for name in FACE_NAMES:
        path = directory/f'{name}_95.png'
        with Image.open(path) as image:
            if image.format != 'PNG' or image.mode != 'RGB' or image.size != (512,512):
                raise ValueError('Require six RGB 512x512 PNG faces')
            faces.append(np.asarray(image,dtype=np.float32)/255)
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = overlap_band_diagnostic(torch.from_numpy(np.stack(faces)).permute(0,3,1,2),
        samples=128,band_samples=17)
    return {'scope':'raw_overlap_diagnostic','objective':'minimize',
        'samples_per_edge':128,'band_samples':17,'edge_count':12,
        'overlap_band':raw,'image_sha256':fingerprints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--faces',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = measure(args.faces)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"Raw overlap RGB L1: {result['overlap_band']['mean_rgb_l1']:.8f}")


if __name__=='__main__':
    main()
