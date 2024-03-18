# from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# feature_extractor = AutoFeatureExtractor.from_pretrained("aTunass/wav2vec2-base-finetuned-ks")
# model = AutoModelForAudioClassification.from_pretrained("aTunass/wav2vec2-base-finetuned-ks")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
tokenizer = AutoTokenizer.from_pretrained("aTunass/distilbert-base-uncased-finetuned-Tweets_IMDB_dataset-finetuned-text_classification")
model = AutoModelForSequenceClassification.from_pretrained("aTunass/distilbert-base-uncased-finetuned-Tweets_IMDB_dataset-finetuned-text_classification",
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)
text = "This movie is too weird"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax().item()
result = model.config.id2label[predicted_class_id]
print(inputs)
print(inputs["input_ids"].shape)
print(predicted_class_id)
print(result)