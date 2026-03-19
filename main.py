import json

from dotenv import load_dotenv
from openai import OpenAI

# create a .env file with an OpenAI key inside
load_dotenv()

# initialize the client
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


# mock data created by an LLM
mock_email_body = """
Hello FinServe team,

My name is Sarah Jenkins and I am the CEO of TechFlow Solutions Ltd. We are looking to 
apply for a line of credit. Our company registration number is 88472910. 

We would like to request $75,000 to help with purchasing new server equipment and 
expanding our marketing team for the Q3 push. For context, our annual revenue last 
year was roughly $450,000. 

Please let me know what the next steps are to get this approved and into your core system.

Best,
Sarah
"""

# run the whole thing using python main.py command
if __name__ == "__main__":
    print("Processing incoming application...")
    print("-" * 40)

    structured_result = extract_application_data(mock_email_body)

    print("Extraction Complete. Outputting structured JSON for Core Banking System:")
    print(json.dumps(structured_result, indent=4))
