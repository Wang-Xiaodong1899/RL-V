import gradio as gr
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

model_path = '/workspace/wxd/RL-V/roberta-large-wanli'
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

def predict(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', max_length=128, truncation=True)
    logits = model(**inputs).logits
    probs = logits.softmax(dim=1).squeeze(0)
    print(probs) # [contradiction, entailment, neutral]
    label_id = torch.argmax(probs).item()
    prediction = model.config.id2label[label_id]
    return prediction

# chosen = "The image depicts a train traveling on a track through a countryside setting with tall grass, trees, and power lines in the background."
# rejected = "The image is set in an open area with train tracks, grassy fields, and trees in the background."

chosen = "cat"
rejected = "animal"

# chosen -> rejected (Entailment relationship)

predict(chosen, rejected)

# iface = gr.Interface(
#     fn=predict,
#     inputs=["text", "text"],
#     outputs="text",
#     title="Roberta Sequence Classification",
#     description="输入两个句子，模型将预测它们之间的关系。"
# )

# iface.launch()
