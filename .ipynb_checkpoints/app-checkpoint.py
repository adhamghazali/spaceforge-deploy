import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image

#st.header('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
st.components.v1.html('', width=None, height=None, scrolling=True)



#st.set_page_config(
#   page_title="Spacetink Demo",
#   page_icon="🧊",
#   layout="wide",
#   initial_sidebar_state="expanded",
#)




def base64_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image

st.markdown('## LEAGUE OF IMAGINATION')
st.markdown("### Today's Challenge: create a futuristic city")

prompt = st.text_input('Start with a detailed description')


if st.button('Generate Image'):
    data=''
    
    with st.spinner('Buring GPUs...'):
        for i in requests.post("http://34.134.112.46:8000/?prompt="+prompt):
            data=data+i.decode()
        image=base64_pil(data)
        st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
     
st.markdown('### Examples:')
st.markdown('A futuristic city finely detailed, intricate design, silver buildings, tiles roads, cinematic lighting, 4k')
