from transformers import BartForConditionalGeneration, BartTokenizer
from Aluora.models.llm_base import LLMModel  # Si lo separas
import torch

class BartCNN(LLMModel):
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        super().__init__()

    def generate(self, inpt):
        inputs = self.tokenizer([inpt], return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs["input_ids"])
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
