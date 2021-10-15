# Importing dependencies from transformers
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# takes some time:
# Load tokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
# Load model
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

def summarize(text):
    """
    Uses the pegasus model (google) to summarize texts
    """
    # Create tokens - number representation of our text
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")

    # Summarize
    summary = model.generate(**tokens)

    # return decoded summary
    return tokenizer.decode(summary[0])
