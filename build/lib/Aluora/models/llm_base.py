from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn.functional as F


class LLMModel:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        pass

    def getName(self) -> str:
        return self.model_name

    def getSanitizedName(self) -> str:
        return self.model_name.replace("/", "__")

    def generate(self, inpt):
        pass

    ##Move in future commits this method to an utils.py
    def truncate_string_by_len(self, s, truncate_len):
        words = s.split()
        truncated_words = words[:-truncate_len] if truncate_len > 0 else words
        return " ".join(truncated_words)

    # Method to get the vocabulary probabilities of the LLM for a given token on the generated text from LLM-Generator
    def getVocabProbsAtPos(self, pos, token_probs):
        sorted_probs, sorted_indices = torch.sort(token_probs[pos, :], descending=True)
        return sorted_probs

    def getMaxLength(self):
        return self.model.config.max_position_embeddings

    # By default knowledge is the empty string. If you want to add extra knowledge you can do it like in the cases of the qa_data.json and dialogue_data.json
    def extractFeatures(
        self,
        knowledge="",
        conditionted_text="",
        generated_text="",
        features_to_extract={},
    ):
        self.model.eval()

        total_len = len(knowledge) + len(conditionted_text) + len(generated_text)
        truncate_len = min(total_len - self.tokenizer.model_max_length, 0)

        # Truncate knowledge in case is too large
        knowledge = self.truncate_string_by_len(knowledge, truncate_len // 2)
        # Truncate text_A in case is too large
        conditionted_text = self.truncate_string_by_len(
            conditionted_text, truncate_len - (truncate_len // 2)
        )

        inputs = self.tokenizer(
            [knowledge + conditionted_text + generated_text],
            return_tensors="pt",
            max_length=self.getMaxLength(),
            truncation=True,
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = F.softmax(logits, dim=-1)
        probs = probs.to(self.device)

        tokens_generated_length = len(self.tokenizer.tokenize(generated_text))
        start_index = logits.shape[1] - tokens_generated_length
        conditional_probs = probs[0, start_index :]

        token_ids_generated = inputs["input_ids"][0, start_index :].tolist()
        token_probs_generated = [
            conditional_probs[i, tid].item()
            for i, tid in enumerate(token_ids_generated)
        ]

        tokens_generated = self.tokenizer.convert_ids_to_tokens(token_ids_generated)
        # Filtrar tokens de fin de secuencia
        EOS_TOKENS = {'</s>', '<EOS>', '<eos>'}
        non_eos = [
            (token, prob)
            for token, prob in zip(tokens_generated, token_probs_generated)
            if token not in EOS_TOKENS
        ]
        if non_eos:
            min_prob_token, min_prob = min(non_eos, key=lambda x: x[1])
        else:
            min_prob_token, min_prob = None, None
        minimum_token_prob = min(token_probs_generated)
        average_token_prob = sum(token_probs_generated) / len(token_probs_generated)

        maximum_diff_with_vocab = -1
        minimum_vocab_extreme_diff = 100000000000

        if features_to_extract["MDVTP"] == True or features_to_extract["MMDVP"] == True:
            size = len(token_probs_generated)
            for pos in range(size):
                vocabProbs = self.getVocabProbsAtPos(pos, conditional_probs)
                maximum_diff_with_vocab = max(
                    [
                        maximum_diff_with_vocab,
                        self.getDiffVocab(vocabProbs, token_probs_generated[pos]),
                    ]
                )
                minimum_vocab_extreme_diff = min(
                    [
                        minimum_vocab_extreme_diff,
                        self.getDiffMaximumWithMinimum(vocabProbs),
                    ]
                )

        allFeatures = {
            "mtp": minimum_token_prob,
            "avgtp": average_token_prob,
            "MDVTP": maximum_diff_with_vocab,
            "MMDVP": minimum_vocab_extreme_diff,
        }

        selectedFeatures = {}
        for key, feature in features_to_extract.items():
            if feature == True:
                selectedFeatures[key] = allFeatures[key]

        return selectedFeatures

    def getDiffVocab(self, vocabProbs, tprob):
        return (vocabProbs[0] - tprob).item()

    def getDiffMaximumWithMinimum(self, vocabProbs):
        return (vocabProbs[0] - vocabProbs[-1]).item()