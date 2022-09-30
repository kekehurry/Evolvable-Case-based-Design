import streamlit as st 
import streamlit.components.v1 as components
import os
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import random
import json,cv2
from model.ddpg import Env,DDPG
import plotly.express as px
import plotly.graph_objects as go

agent = DDPG()
env = Env()

st.set_page_config(
     page_title="Evolvable Case-based Design",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "Design Future Lab\n Contant:837082742@qq.com "
     }
 )


st.title("Evolvable Case-based Design")
st.text("Genrate City Morphology with specific FSI, GSI and road system")
placeholder1 = st.empty()
placeholder2 = st.empty()
# col1,col2 = st.columns([4,2])
# with col1:
#     placeholder2 = st.empty()
# with col2:
#     placeholder3 = st.empty()

st.session_state.fsi_coeff = 0.
st.session_state.gsi_coeff = 0.
st.session_state.l_coeff = 0.
st.session_state.osr_coeff = 0.

#generate
def generate(input,target_FSI,target_GSI,max_iter):
    env.seg = env.path2seg(input)
    env.target_FSI = target_FSI
    env.target_GSI = target_GSI
    state = env.reset()
    action_history = []
    reward_history = []
    p=placeholder1.progress(0)
    for i in range(max_iter):
        p.progress(i/(max_iter-1))
        action = agent.select_action(state)
        action = (action + np.random.normal(0, agent.args.exploration_noise, size=action.shape)).clip(
                    -agent.args.max_action, agent.args.max_action)
        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.push((state, next_state, action, reward, float(done)))
        action_history.append(action)
        reward_history.append(reward)
        state = next_state
        agent.update()
    idx = np.argmax(reward_history)
    action = action_history[idx]
    img,latent_vector = env.generate_image(action)
    data,FSI,GSI,L,OSR = env.createmodel(img)
    placeholder1.empty()
    return data,FSI,GSI,L,OSR,latent_vector


#visualizer
def visualizer(FSI,GSI,L,OSR,target_FSI,target_GSI):
    target_L = target_FSI/target_GSI
    target_OSR = (1-target_GSI)/target_FSI
    target_values = [target_FSI,target_GSI,target_L,target_OSR]
    max_values = [4,1,20,1]
    values = [FSI,GSI,L,OSR]
    labels = ['FSI','GSI','L', 'OSR']
        
    with open('templates/index.html','r') as f:
        html = f.read()
    with placeholder2.container():
        components.html(html.replace(r'{{data}}',data), height=600)

    # with placeholder3.container():
    #     st.markdown('<br></br><br></br>',unsafe_allow_html=True)
    #     df1 = pd.DataFrame(dict(
    #         r=np.array(target_values+values)/np.array(max_values+max_values),
    #         theta=labels+labels,
    #         type = np.array(['target','target','target','target','result','result','result','result'])))
    #     fig1 = px.line_polar(df1, r='r', theta='theta',color='type',line_close=True)
    #     fig1.update_traces(fill='toself')
    #     fig1.update_layout(
    #         polar=dict(
    #             radialaxis=dict(
    #             range = [0,1],
    #             visible=True
    #             ),
    #         ),
    #         showlegend=False,
    #         )
    #     st.plotly_chart(fig1, use_container_width=True)

    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.number_input(labels[i],value=values[i])


st.sidebar.caption("INPUT PARAMETERS")
form1 = st.sidebar.form('Form1')
option = form1.selectbox("EXAMPLE INPUT",('example_1','example_2','example_3'))
input1 = form1.file_uploader('INPUT')
if input1:
    input = input1
else:
    with open('static/%s.png'%option,'rb') as f:
        input = f.read()
    input = BytesIO(input)
form1.image(input)
# target
target_FSI=form1.slider('TARGET FSI',0.,5.0,2.6,0.1)
target_GSI=form1.slider('TARGET GSI',0.,0.5,0.36,0.01)
seed = form1.slider('SEED',0,100,50,1)
max_iter=form1.slider('MAX ITER',10,100,50,10)
gen_button = form1.form_submit_button('generate')
if gen_button:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    data,FSI,GSI,L,OSR,init_latent_vector = generate(input,target_FSI,target_GSI,max_iter)
    st.session_state.init_latent_vector = init_latent_vector
    st.session_state.input = input
    visualizer(FSI,GSI,L,OSR,target_FSI,target_GSI)

# adjust
st.sidebar.caption("ADJUST PARAMETERS")
form2 = st.sidebar.form('Form2',clear_on_submit=True)
fsi_coeff = form2.slider('FSI DIRECTION',-1.,1.,0.,step=0.01)
gsi_coeff = form2.slider('GSI DIRECTION',-1.,1.,0.,step=0.01)*3
l_coeff = form2.slider('L DIRECTION',-1.,1.,0.,step=0.01)
osr_coeff = form2.slider('OSR DIRECTION',-1.,1.,0.,step=0.01)
col5,col6 = form2.columns(2)
adj_button = form2.form_submit_button('adjust')
if adj_button:
    env.seg = env.path2seg(st.session_state.input)
    init_latent_vector = st.session_state.init_latent_vector
    new_img,latent_vector = env.adjust_image(init_latent_vector,env.fsi_direction,fsi_coeff)
    new_img,latent_vector = env.adjust_image(latent_vector,env.gsi_direction,gsi_coeff)
    new_img,latent_vector = env.adjust_image(latent_vector,env.l_direction,l_coeff)
    new_img,latent_vector = env.adjust_image(latent_vector,env.osr_direction,osr_coeff)
    st.session_state.init_latent_vector = latent_vector
    data,FSI,GSI,L,OSR = env.createmodel(new_img)
    visualizer(FSI,GSI,L,OSR,target_FSI,target_GSI)

