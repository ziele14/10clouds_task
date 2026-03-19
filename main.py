import json

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize the client
client = OpenAI()


def extract_application_data(unstructured_text):
    """
    Uses an LLM to extract key loan application data from unstructured text
    and returns it as a structured JSON object.
    """
    system_prompt = """
    You are an AI assistant for FinServe, a financial services company. 
    Your job is to extract loan application data from raw text emails and portal submissions.
    Extract the following fields and return ONLY a valid JSON object:
    - applicant_name (string)
    - company_name (string)
    - company_registration_number (string)
    - loan_amount_requested (integer)
    - loan_purpose (string)
    - annual_revenue (integer)
    
    If a field is missing, set its value to null.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": unstructured_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # low temperature for more deterministic, factual extraction
        )

        # parse the JSON string returned by the model
        extracted_data = json.loads(response.choices[0].message.content)
        return extracted_data

    except Exception as e:
        return {"error": str(e)}


# quick Streamlit app
st.set_page_config(page_title="FinServe Data Extractor", page_icon="🏦", layout="wide")

st.title("FinServe: Automated Application Extractor")
st.markdown(
    "This tool reads unstructured emails and portal submissions, automatically extracting the required data into JSON for our Core Banking System."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Incoming Application (Email/Portal)")

    mock_email_body = """Hello FinServe team,

My name is Sarah Jenkins and I am the CEO of TechFlow Solutions Ltd. We are looking to 
apply for a line of credit. Our company registration number is 88472910. 

We would like to request $75,000 to help with purchasing new server equipment and 
expanding our marketing team for the Q3 push. For context, our annual revenue last 
year was roughly $450,000. 

Please let me know what the next steps are to get this approved and into your core system.

Best,
Sarah"""

    user_input = st.text_area(
        "Paste unstructured text here:", value=mock_email_body, height=300
    )
    process_button = st.button("Extract Data", type="primary")

with col2:
    st.subheader("Structured Output (JSON)")

    if process_button:
        with st.spinner("AI is analyzing the text..."):
            structured_result = extract_application_data(user_input)

            if "error" in structured_result:
                st.error(f"Extraction failed: {structured_result['error']}")
            else:
                st.success("Extraction Complete!")
                st.json(structured_result)
