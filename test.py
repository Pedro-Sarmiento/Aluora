from Aluora.core.extractor import hallucination_metrics

context = "Paris is the capital of France."
question = "What is the capital of France?"
answer = "The capital city of France is Paris."

scores = hallucination_metrics(context, question, answer)


