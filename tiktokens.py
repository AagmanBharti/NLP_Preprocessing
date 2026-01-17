import tiktoken
text = tiktoken.encoding_for_model("gpt-3.5-turbo")
token = text.encode("Hello world!")
print(token)
print(text.decode(token))
print(len(token))