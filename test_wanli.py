import gradio as gr
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

model_path = '/mnt/storage/user/wangxiaodong/RLAIF-V/roberta-large-wanli'
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

def predict(sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', max_length=128, truncation=True)
    logits = model(**inputs).logits
    probs = logits.softmax(dim=1).squeeze(0)
    label_id = torch.argmax(probs).item()
    prediction = model.config.id2label[label_id]
    return prediction

iface = gr.Interface(
    fn=predict,
    inputs=["text", "text"],
    outputs="text",
    title="Roberta Sequence Classification",
    description="输入两个句子，模型将预测它们之间的关系。"
)

iface.launch()
