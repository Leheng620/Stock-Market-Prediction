import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.roberta_based import RobertaLSTM
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = RobertaLSTM(sentiment_model, time_window=5)

def preprocess_dataset(batch):
    # print(len(batch["text"]))
    input_tensors = []
    mask_tensors = []
    for text in batch["text"]:
        tweet = tokenizer(text, return_tensors="pt", padding="max_length",truncation=True,max_length=256)
        input_ids, attention_mask = tweet["input_ids"], tweet["attention_mask"]
        input_tensors.append(input_ids)
        mask_tensors.append(attention_mask)
    # print(len(batch["text"]))
    input_ids, attention_mask = torch.stack(input_tensors, dim=0), torch.stack(mask_tensors, dim=0)
    # print(input_ids.shape)
    prices = torch.tensor(batch["open"]).unsqueeze(-1).float()
    volumes = torch.tensor(batch["volume"]).unsqueeze(-1).float()
    labels = torch.tensor(batch["close"]).float()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "prices": prices, "volumes": volumes, "labels": labels}


def custom_collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    prices = [torch.tensor(item["prices"]) for item in batch]
    volumes = [torch.tensor(item["volumes"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    prices = torch.nn.utils.rnn.pad_sequence(prices, batch_first=True, padding_value=0)
    volumes = torch.nn.utils.rnn.pad_sequence(volumes, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "prices": prices, "volumes": volumes, "labels": labels}






time_window = 5
batch_size = 8

hf_dataset = load_dataset("json", data_files={"train": "/Users/pan/Documents/course/EECS545/Group/Stock-Market-Prediction/data/tweet_price/aligned_data.json"}, split="train")
hf_dataset = hf_dataset.map(preprocess_dataset, batched=True)
train_val_split = hf_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)





device = torch.device("mps")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs in dataloader:
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prices = inputs["prices"].to(device)
        volumes = inputs["volumes"].to(device)
        labels = inputs["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, prices, volumes)
        loss = criterion(outputs[:, -1], labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_predictions = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs in dataloader:
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            prices = inputs["prices"].to(device)
            volumes = inputs["volumes"].to(device)
            labels = inputs["labels"].to(device)



            outputs = model(input_ids, attention_mask, prices, volumes)
            loss = criterion(outputs[:, -1], labels)

            running_loss += loss.item()

            # Calculate accuracy
            predictions = (outputs[:, -1] > 0).int()
            total_predictions += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()

    return running_loss / len(dataloader), correct_predictions / total_predictions



num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
