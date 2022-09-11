import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
import time



import datetime
from datetime import date

today = datetime.date.today()
yourday = datetime.date(2022, 8, 27)
two = datetime.date(2022, 8, 26)

timebeetwen = today - yourday
timevetween2=today-two
counter=timebeetwen.days*24
counter2=timevetween2.days*24



#st.header('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
st.components.v1.html('', width=None, height=None, scrolling=True)



#st.set_page_config(
#   page_title="Spacetink Demo",
#   page_icon="🧊",
#   layout="wide",
#   initial_sidebar_state="expanded",
#)

from streamlit.components.v1 import html





def base64_pil(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image

st.markdown('## LEAGUE OF IMAGINATION')
st.markdown("### Today's Challenge: Animals & Vehicles")
#st.markdown('### Number of active participants: 1008'#+(str(int(counter+5*(counter2-counter)+6*(counter-20)))))

prompt = st.text_input('Start with a detailed description')


if st.button('Generate Image'):
    data=''
    
    with st.spinner('Buring GPUs...'):
        
        try:
            time.sleep(1)
            for i in requests.post("http://34.134.112.46:8000/p0?prompt="+prompt):
                data=data+i.decode()
            image=base64_pil(data)
            st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        except:
            try:
                time.sleep(1)

                for i in requests.post("http://34.134.112.46:8000/p1?prompt="+prompt):
                    data=data+i.decode()
                image=base64_pil(data)
                st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
            except:
                try:
                    time.sleep(1)

                    for i in requests.post("http://34.134.112.46:8000/p2?prompt="+prompt):
                        data=data+i.decode()
                    image=base64_pil(data)
                    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
                    
                except:
                    st.markdown('try again later')
            
     
st.markdown('### Examples:')
st.markdown('A cat driving a motorcycle, 8k, artstation')
st.markdown('A panda riding large container ship, 8k, 3d art')



# Define your javascript
my_js = """
;
"""

# Wrapt the javascript as html code
my_html = f"<script>{my_js}</script>"

html(my_html)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


