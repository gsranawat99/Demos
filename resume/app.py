from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st
from PIL import Image
import base64
import io
import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_response(input, pdf_content , prompt):
    model= genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    response=model.generate_content([input, pdf_content[0],prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    #convert pdf to image
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]

        #convert to bytes
        img_bytes_arr=io.BytesIO()
        first_page.save(img_bytes_arr, format='JPEG')
        img_bytes_arr=img_bytes_arr.getvalue()

        pdf_parts=[
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_bytes_arr).decode('utf-8') #encode to base64
            }
        ]
        return pdf_parts
    else:
        return FileNotFoundError("No file uploaded")
    
## Streamlit APP

st.set_page_config(page_title="ATS Resume Scanner", page_icon=":guardsman:", layout="wide") 
st.title("ATS Resume Scanner")
input_text=st.text_area("Enter your job description here:", key="input",height=200)
uploaded_file = st.file_uploader("Upload your resume here:", type=["pdf"], key="file_uploader")

if uploaded_file is not None:
    st.write("PDF file uploaded successfully.")

submit1=st.button("tell me about my resume")
# submit2=st.button("How can i improve my resume")
# submit3=st.button("what are keywords that are missing?")
submit4=st.button("percentage match")


input_prompt1 = """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt4 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""  

if submit1:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("The Repsonse is")
        st.write(response)
    else:
        st.write("Please uplaod the resume")

elif submit4:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt4,pdf_content,input_text)
        st.subheader("The Repsonse is")
        st.write(response)
    else:
        st.write("Please uplaod the resume")
