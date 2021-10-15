import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

def create_text(sentence, max_length = 50):
    # tokenize the input
    input_ids = tokenizer.encode(sentence, return_tensors='pt')

    # generate text until the output length (which includes the context length) reaches 50
    output = model.generate(input_ids, max_length = max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

    return tokenizer.decode(output[0], skip_special_tokens=True)
