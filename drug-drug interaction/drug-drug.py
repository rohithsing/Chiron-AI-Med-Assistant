import csv
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def load_drug_interaction_data():
    interactions = []
    for filename in ['ddinter_downloads_code_A.csv', 'ddinter_downloads_code_R.csv', 'ddinter_downloads_code_V.csv']:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Level'] == 'Major':  # Only collect major interactions
                    interactions.append(row)
    return interactions

def analyze_alternative_medications(drug_a, drug_b, interaction_level, model="mixtral-8x7b-32768"):
    prompt = f"""Analyze the following drug interaction and provide detailed information about alternative medications:

Drug A: {drug_a}
Drug B: {drug_b}
Interaction Level: {interaction_level}

Please provide:
1. Brief overview of why these drugs interact
2. Detailed alternative medications for Drug A:
   - List safer alternatives
   - Explain why they're better choices
   - Note any precautions with alternatives
3. Detailed alternative medications for Drug B:
   - List safer alternatives
   - Explain why they're better choices
   - Note any precautions with alternatives
4. Best practices for switching to alternatives
5. Important monitoring requirements when switching medications

Focus on providing practical, evidence-based alternatives that minimize drug interactions while maintaining therapeutic effectiveness."""

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.3,
        max_tokens=1000
    )
    
    return completion.choices[0].message.content

def main():
    # Load the data
    interactions = load_drug_interaction_data()
    
    print("Analyzing alternative medications for major drug interactions...\n")
    
    # Analyze first 3 major interactions
    for i, interaction in enumerate(interactions[:3]):
        print(f"\nInteraction {i+1}:")
        print(f"Finding alternatives for {interaction['Drug_A']} and {interaction['Drug_B']}...")
        analysis = analyze_alternative_medications(
            interaction['Drug_A'],
            interaction['Drug_B'],
            interaction['Level']
        )
        print("\nAnalysis:")
        print(analysis)
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
