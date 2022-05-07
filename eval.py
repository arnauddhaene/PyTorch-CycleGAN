from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st


st.set_page_config(layout="wide")

org_path = Path('datasets/taggedmr-v1/test/')
syn_path = Path('output/')

# way = st.selectbox('Direction', ('AtoB', 'BtoA'))
way = 'AtoB'

if way == 'AtoB':
    org_path = org_path / 'A'
    syn_path = syn_path / 'B'
else:
    org_path = org_path / 'B'
    syn_path = syn_path / 'A'

orgs = filter(lambda p: p.is_file(), org_path.iterdir())
syns = filter(lambda p: p.is_file(), syn_path.iterdir())
orgs, syns = sorted(orgs, key=lambda p: p.stem), sorted(syns, key=lambda p: p.stem)

masks = filter(lambda p: p.is_file(), (org_path / 'masks').iterdir())
masks = sorted(masks, key=lambda p: p.stem)

f_tagged = filter(lambda p: p.is_file(), (org_path / 'fake_tagged').iterdir())
f_tagged = sorted(f_tagged, key=lambda p: p.stem)

idx = st.slider('Select image', 0, len(orgs))

img_org = Image.open(orgs[idx]).convert('L')
img_syn = Image.open(syns[idx]).convert('L')
mask = np.load(masks[idx])['label']
f_tag_org = Image.open(f_tagged[idx]).convert('L')

fig, ax = plt.subplots(1, 3)

ax[0].imshow(img_org, cmap='gray')
ax[0].imshow(mask, cmap='Reds', alpha=0.3)
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(f_tag_org, cmap='gray')
ax[1].imshow(mask, cmap='Reds', alpha=0.3)
ax[1].axis('off')
ax[1].set_title('Fake tagged')

ax[2].imshow(img_syn, cmap='gray')
ax[2].imshow(mask, cmap='Reds', alpha=0.3)
ax[2].axis('off')
ax[2].set_title('GAN tagged')

st.pyplot(fig)
