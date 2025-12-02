import json
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import re
import os
# Preprocessing Function
def preprocess_tamil_text(text):
    text = re.sub(r"[^ஂ-ஔஎ-௺\s]", "", text)  # Retain Tamil script and spaces only
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Train SentencePiece tokenizer
def train_tokenizer(corpus_file, model_prefix="tamil_tokenizer", vocab_size=32000):
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe"
    )
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

# Dataset Class for QA Data (training dataset)
class TamilQADataset(Dataset):
    def _init_(self, qa_data, tokenizer, seq_length):
        self.data = qa_data
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        item = self.data[idx]
        question = preprocess_tamil_text(item["question"])
        answer = preprocess_tamil_text(item["answers"][0])  # Assume the first answer is used

        question_tokens = self.tokenizer.encode(question, out_type=int)
        answer_tokens = self.tokenizer.encode(answer, out_type=int)

        # Combine question and answer with a separator token
        sep_token = self.tokenizer.piece_to_id("</s>")  # Default SEP token in SentencePiece
        tokens = question_tokens + [sep_token] + answer_tokens

        # Ensure sequence length constraints
        tokens = tokens[:self.seq_length]
        src = torch.tensor(tokens, dtype=torch.long)

        # Generate labels for start and end positions
        start_idx = len(question_tokens) + 1  # Start of the answer
        end_idx = start_idx + len(answer_tokens) - 1  # End of the answer
        start_label = torch.tensor(min(start_idx, self.seq_length - 1), dtype=torch.long)
        end_label = torch.tensor(min(end_idx, self.seq_length - 1), dtype=torch.long)

        return src, start_label, end_label

# Transformer-based QA Model
class TamilQAModel(nn.Module):
    def _init_(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TamilQAModel, self)._init_()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_length, d_model)
        )
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward
        )
        self.start_logits = nn.Linear(d_model, 1)
        self.end_logits = nn.Linear(d_model, 1)

    def forward(self, input_tokens):
        embeddings = self.embedding(input_tokens) + self.positional_encoding[:, :input_tokens.size(1), :]
        transformer_output = self.transformer(embeddings, embeddings)
        start_logits = self.start_logits(transformer_output).squeeze(-1)
        end_logits = self.end_logits(transformer_output).squeeze(-1)
        return start_logits, end_logits

# Training Loop
def train_model(model, dataloader, optimizer, loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, start_label, end_label in dataloader:
            optimizer.zero_grad()
            start_logits, end_logits = model(src)

            # Calculate losses
            loss_start = loss_fn(start_logits, start_label)
            loss_end = loss_fn(end_logits, end_label)
            loss = (loss_start + loss_end) / 2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Main Workflow
if __name__ == "__main__":
    # Set paths for your data and corpus
    qa_data_file = r"C:\nlp\QA data file"  # Example
    test_data_file = r"C:\nlp\test_data_file"
    corpus_file = "C:\nlp\corpus.txt"  # Path to corpus file for training tokenizer

    vocab_size = 32000  # Adjusted vocabulary size
    seq_length = 512  # Maximum sequence length

    # Load QA data
    with open(qa_data_file, encoding="utf-8") as file:
        qa_data = json.load(file)

    # Load test data (words for testing)
    with open(test_data_file, encoding="utf-8") as file:
        test_data = json.load(file)

    # Save data to a text file for tokenizer training
    texts = [item["question"] + " " + item["answers"][0] for item in qa_data]  # Update to reflect actual structure
    with open(corpus_file, "w", encoding="utf-8") as file:
        file.write("\n".join(texts))

    # Train the tokenizer
    tokenizer = train_tokenizer(corpus_file, vocab_size=vocab_size)

    # Initialize dataset and dataloader for training
    dataset = TamilQADataset(qa_data, tokenizer, seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model, optimizer, and loss function
    model = TamilQAModel(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=seq_length
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Train model
    train_model(model, dataloader, optimizer, loss_fn, epochs=10)

    # Now, we can use the trained model for testing using the test.json data
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for word in test_data:
            word = preprocess_tamil_text(word)  # Preprocess the word
            tokens = tokenizer.encode(word, out_type=int)
            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
            start_logits, end_logits = model(tokens_tensor)
            # You can use start_logits and end_logits for testing or prediction
            print(f"Predicted start and end for '{word}': Start: {start_logits}, End: {end_logits}")
            
