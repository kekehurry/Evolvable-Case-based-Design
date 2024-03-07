import os,io
from PIL import Image
from util.solver import Solver
from util.parser import get_parser
from torchvision import transforms

import streamlit as st


def generate(solver,img,seg):
    seg = solver.dataset['train'].encode_segmap(seg)
    seg = solver.dataset['train'].transform(Image.fromarray(seg))
    img = solver.dataset['train'].transform(img)

    img = img.unsqueeze(0).to(solver.device)
    seg = seg.unsqueeze(0).to(solver.device)
    mu , logvar = solver.nets.encoder(img)
    z = solver.reparameterize( mu, logvar)
    gen_img = solver.nets.generator(seg, z)
    gen_img = transforms.ToPILImage()(gen_img[0])
    return gen_img

def img2bytes(img):
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def bytes2img(bytes):
    imgtemp = io.BytesIO(bytes)
    img = Image.open(imgtemp).convert('RGB')
    return img

st.title('Test Generator')
parser = get_parser()
args = parser.parse_args(args=[])

st.sidebar.caption('setting for args')
args.name = st.sidebar.text_input(label='name', value = r'City')
args.data_dir = st.sidebar.text_input(label='data_dir', value = r'datasets/City')
args.color_file= st.sidebar.text_input(label='color_file', value = r'data/CityColor.txt')
args.dataset_mode = st.sidebar.selectbox('dataset_mode',['LabelDataset','PairDataset'])
latest = st.sidebar.radio(label='load_latest',options=['yes','no'])
placeholder = st.sidebar.empty()
if latest == 'no':
    resume_epoch= placeholder.number_input(label='resume_epoch', value = 300,step=20)
else:
    resume_epoch = 0

col1,col2= st.columns(2)
with col1:
    style_path=st.file_uploader(label='Style Image')
with col2:
    seg_path = st.file_uploader(label='Segmetic Image')
    
col4,col5,col6 = st.columns(3)
with col4:
    if style_path:
        st.image(style_path,use_column_width=True)
with col5:
    if seg_path:
        st.image(seg_path,use_column_width=True)
with col6:
    placeholder1 = st.empty()
       
if style_path and seg_path:
    style_img = bytes2img(style_path.getvalue())
    seg_img = bytes2img(seg_path.getvalue())
    solver = Solver(args)
    if resume_epoch !=0:
        solver.load_model(resume_epoch)
    else:
        solver.load_model(latest=True)
    gen_img = generate(solver,style_img,seg_img)
    placeholder1.image(img2bytes(gen_img),use_column_width=True)