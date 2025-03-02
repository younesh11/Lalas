import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class TextClassifier:
    def __init__(self, model_path):
       
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):

        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()} 

        # Perform inference
        with torch.no_grad(): 
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        predicted_class = torch.argmax(logits, dim=-1).item()

        return predicted_class, probabilities