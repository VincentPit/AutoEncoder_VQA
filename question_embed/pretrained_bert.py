from transformers import BertTokenizer, BertModel
import torch

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def easy_bert(question: str ):
    
    inputs = tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    # question_representation
    return hidden_states[:, :, :]


if __name__ == "__main__":
    print((easy_bert( "What is the capital of France please tell me")).shape)
