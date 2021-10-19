

# Import the model class and the tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration



# Download and setup the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")


def conv_reply(utterance):
    # Tokenize the utterance
    inputs = tokenizer(utterance, return_tensors="pt")
    # Passing through the utterances to the Blenderbot model
    res = model.generate(**inputs)
    # Decoding the model output
    return tokenizer.decode(res[0])
