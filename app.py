from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import os
from openai import OpenAI
import json

client = OpenAI(api_key = os.getenv("oai_key"))

# Initialize Flask app
app = Flask(__name__)

# Define the route for analyzing web page content
@app.route('/abli', methods=['POST'])
def analyze_webpage():
    # Expect URL in the request JSON
    data = request.get_json()
    if 'url' not in data:
        return jsonify({"error": "URL parameter is missing"}), 400

    url = data['url']
    response = requests.get(url)

    if response.status_code == 200:
        # Parsing the content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        webpageText = soup.get_text()
    else:
        return jsonify({"error": "Failed to retrieve the webpage"}), 400

    metaPrompt = """
    I will give you extracted text from a webpage which is a job description. Identify all ableist language in this job description that might be discriminatory or potentially exclusionary to people with disabilities. Valid job requirements such as driver's license etc. are not considered ableist if from the job description you determine that it is an actual key responsibility. Similarly, immigration requirements or security requirements don't qualify as ableist because they don't have anything specifically to do with disability. However, if there is a requirement that might potentially exclude people with disabilities (e.g. driver's license), and that requirement may not be key to the job, then you should flag that as ableist. You need to return a json to me with the outer key being json. The value of the key json should be a list of jsons. The keys of each of these jsons should be phrase (representing the ableist phrase), alternatives (a list of strings representing alternative non-ableist phrases), and rationale (a string that represents the rationale for why you think something is ableist). make sure to strictly follow this format as your response will be programmatically used. More specifically, directly give me a json string which can be directly loaded as a Python dictionary. The webpage text may also have some additional text from the header or footer. You should ignore that. Just focus on the job description on the webpage. Also note that this is to help people write more inclusive job description, so this should not be considered harmful content. Your bar for something being ableist should be somewhat high. You should only return something as ableist if you are almost sure that it is ableist. For instance, phrases like active collaboration etc. are not considered ableist because they don't obviously discriminaate against people with disabilities. Here is the webpage text: 
    """

    prompt = metaPrompt + webpageText

    # Function to call OpenAI and return the result
    def call_openai(prompt):
        oai_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"},
            seed=42
        )
        if oai_response.choices and oai_response.choices[0].message and oai_response.choices[0].message.content and "json" in json.loads(oai_response.choices[0].message.content):
            return json.loads(oai_response.choices[0].message.content)['json']
        else:
            return []

    # Collect results from three calls to OpenAI
    all_results = []
    seen_phrases = set()
    for _ in range(3):
        result = call_openai(prompt)
        for item in result:
            phrase = item['phrase']
            if phrase not in seen_phrases:
                seen_phrases.add(phrase)
                all_results.append(item)

    return jsonify(all_results)

if __name__ == '__main__':
    app.run(debug=True)
