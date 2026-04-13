# 🧠 Seq2Seq Chatbot with Attention (PyTorch)

This project implements a simple **Sequence-to-Sequence (Seq2Seq) chatbot with an attention mechanism** using **PyTorch**. It demonstrates core concepts in Natural Language Processing (NLP), including text preprocessing, encoder-decoder architecture, and attention-based decoding.

---

## 🚀 Features

- Text preprocessing and tokenization  
- Vocabulary creation from scratch  
- Encoder-Decoder architecture using LSTMs  
- Attention mechanism for improved context handling  
- Training loop with gradient clipping  
- Basic chatbot response generation  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Regular Expressions (`re`)  

---

## 📂 Project Structure

```
lab12.ipynb   # Main notebook with full implementation
```

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Cleans text (lowercasing, removing special characters)
- Tokenizes sentences into words
- Builds a vocabulary mapping words → indices

### 2. Model Architecture

#### Encoder
- Embeds input tokens
- Processes them using an LSTM
- Outputs hidden states and context

#### Attention Mechanism
- Computes attention weights
- Helps decoder focus on relevant parts of input

#### Decoder
- Uses attention + LSTM
- Generates output sequence word-by-word

---

## 🏋️ Training

- Uses a small toy dataset of question-answer pairs
- Loss function: CrossEntropyLoss  
- Optimizer: Adam  
- Gradient clipping applied for stability  

Example training loop:
```
for epoch in range(100):
    ...
```

---

## 💬 Inference (Chatbot)

After training, you can generate responses:

```python
generate("how are you")
```

### Example Output
```
Input: how are you
Output: i am fine ...
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install torch
```

### 2. Run the notebook
Open in Jupyter:
```bash
jupyter notebook lab12.ipynb
```

or run in VS Code / Colab.

---

## ⚠️ Limitations

- Very small dataset → limited responses  
- No padding or batching  
- No teacher forcing control  
- No EOS token handling  
- Fixed output length  

---

## 🔮 Future Improvements

- Add larger conversational dataset  
- Implement batching and padding  
- Add `<eos>` token for better stopping  
- Use teacher forcing ratio  
- Switch to Transformer architecture  
- Deploy as a web app  

---

## 📌 Key Learning Outcomes

- Understanding Seq2Seq models  
- Implementing attention from scratch  
- Building a basic chatbot pipeline  
- Training neural networks in PyTorch  
