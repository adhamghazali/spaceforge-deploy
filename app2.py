import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
from streamlit.components.v1 import html
import datetime
from datetime import date
import spaceforge


@st.cache(hash_funcs=None)
def load_model():
    model=spaceforge.Space()
    return model



def main():
    model=load_model()

    st.markdown('## LEAGUE OF IMAGINATION')
    st.markdown("### Today's Challenge: Animals & Vehicles")
    #st.markdown('### Number of active participants: '+(str(int(counter+5*(counter2-counter)+6*(counter-20)))))

    prompt = st.text_input('Start with a detailed description')


    if st.button('Generate Image'):
        data=''

        with st.spinner('Buring GPUs...'):
            image=model.infer(prompt)
            image

        
        
        

    st.markdown('### Examples:')
    st.markdown('A cat driving a motorcycle, 8k, artstation')
    st.markdown('A panda riding large container ship, 8k, 3d art')

    
main()