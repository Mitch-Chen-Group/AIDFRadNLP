from openai import OpenAI

# Replace with your actual key
api_key = "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"

client = OpenAI(api_key=api_key)

# Test a simple call
response = client.models.list()  # lists available models
print(response)