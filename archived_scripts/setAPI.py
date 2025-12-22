from openai import OpenAI

# Replace with your actual key
api_key = ""

client = OpenAI(api_key=api_key)

# Test a simple call
response = client.models.list()  # lists available models
print(response)
