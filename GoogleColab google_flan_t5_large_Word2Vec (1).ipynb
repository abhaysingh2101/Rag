!pip install mlflow faiss-cpu transformers torch sentence-transformers pandas numpy scikit-learn tqdm
import pandas as pd
import numpy as np
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from google.colab import files
from difflib import SequenceMatcher
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm.auto import tqdm

# --- Define the missing compute_additional_metrics function ---
def compute_additional_metrics(true_label, retrieved_labels, distances, k):
    # Top-k Accuracy: Check if true label is in top-k retrieved labels
    top_k_correct = 1 if true_label in retrieved_labels[:k] else 0

    # Mean Reciprocal Rank (MRR): Compute reciprocal rank of first correct label
    mrr = 0
    for rank, label in enumerate(retrieved_labels[:k], 1):
        if label == true_label:
            mrr = 1 / rank
            break

    # Mean Average Precision (MAP): Compute precision at each relevant rank
    relevant_count = 0
    precision_sum = 0
    for rank, label in enumerate(retrieved_labels[:k], 1):
        if label == true_label:
            relevant_count += 1
            precision_sum += relevant_count / rank
    map_score = precision_sum / max(1, relevant_count) if relevant_count > 0 else 0

    # Average Cosine Similarity: Mean of cosine similarities for top-k
    avg_cosine_sim = np.mean(distances[:k])

    return top_k_correct, mrr, map_score, avg_cosine_sim

# --- Upload and load the embedding and sequence files ---
print("Upload 'Malware_api_call_embeddings_wordvec.csv' and 'MalBehavD-V1-dataset.csv'")
uploaded = files.upload()

embedding_df = pd.read_csv('Malware_api_call_embeddings_wordvec.csv')
sequence_df = pd.read_csv('MalBehavD-V1-dataset.csv')

# --- Build api_sequence column from sequence file (joining API call columns) ---
def join_apis(row):
    return ' '.join(str(x) for x in row[2:] if pd.notnull(x))

sequence_df['api_sequence'] = sequence_df.apply(join_apis, axis=1)

# --- Merge sequences into embedding dataframe ---
embedding_df['api_sequence'] = sequence_df['api_sequence']

# --- Extract embeddings, labels, sequences ---
embeddings = embedding_df[[f'emb_{i}' for i in range(50)]].values.astype(np.float32)
labels = embedding_df['labels'].values.astype(np.int32)
api_sequences = embedding_df['api_sequence'].values

# --- Data augmentation with tqdm ---
augmented_data, augmented_labels, sequence_list = [], [], []
np.random.seed(42)

print("Performing data augmentation...")
for i, (emb, label) in tqdm(enumerate(zip(embeddings, labels)), total=len(embeddings)):
    augmented_data.append(emb)
    augmented_labels.append(label)
    sequence_list.append(api_sequences[i])
    for _ in range(5):
        noise_mask = np.random.choice([0, 1], size=50, p=[0.2, 0.8])
        noise = np.random.normal(0, 0.01, size=50) * noise_mask
        perturbed_emb = emb + noise
        augmented_data.append(perturbed_emb)
        augmented_labels.append(label)
        sequence_list.append(api_sequences[i])

augmented_data = np.array(augmented_data, dtype=np.float32)
augmented_labels = np.array(augmented_labels, dtype=np.int32)

# --- Train/test split ---
X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
    augmented_data, augmented_labels, sequence_list, test_size=0.2, random_state=42, stratify=augmented_labels
)

# --- FAISS index ---
dimension = 50
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(X_train)
index.add(X_train)

# --- Load Hugging Face model and tokenizer ---
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Function to generate LLM embeddings ---
def get_llm_embeddings(sequences, batch_size=16):
    embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generating LLM embeddings"):
        batch = sequences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.encoder(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)

# --- Generate LLM embeddings for training and test sequences ---
print("Generating LLM embeddings for training and test sets...")
train_llm_emb = get_llm_embeddings(seq_train)
test_llm_emb = get_llm_embeddings(seq_test)

# --- Function to combine LLM embeddings with FAISS context ---
def combine_features(embeddings, llm_emb, indices, labels, k=5):
    combined_features = []
    for i, (llm_vec, idxs) in enumerate(zip(llm_emb, indices)):
        # Get mean label and cosine similarity of retrieved sequences
        retrieved_labels = [labels[idx] for idx in idxs[:k]]
        mean_label = np.mean(retrieved_labels)
        # Combine LLM embedding with context features
        combined = np.concatenate([llm_vec, [mean_label]])
        combined_features.append(combined)
    return np.array(combined_features)

# --- Train logistic regression classifier ---
print("Training logistic regression classifier...")
k = 5
D_train, I_train = index.search(X_train, k)
train_features = combine_features(X_train, train_llm_emb, I_train, y_train, k)
classifier = LogisticRegression(random_state=42, max_iter=1000)
classifier.fit(train_features, y_train)

# --- Predict using classifier with tqdm for test set evaluation ---
predicted_labels = []
true_labels = []
top_k_corrects = []
mrr_scores = []
map_scores = []
cosine_sims = []

print("Performing prediction on test set...")
D_test, I_test = index.search(X_test, k)
test_features = combine_features(X_test, test_llm_emb, I_test, y_train, k)

for i, (features, true_label, distances, indices) in tqdm(enumerate(zip(test_features, y_test, D_test, I_test)), total=len(y_test)):
    predicted_label = classifier.predict([features])[0]
    retrieved_labels = [y_train[idx] for idx in indices]

    # Compute additional metrics
    top_k_correct, mrr, map_score, avg_cosine_sim = compute_additional_metrics(true_label, retrieved_labels, distances, k)

    predicted_labels.append(predicted_label)
    true_labels.append(true_label)
    top_k_corrects.append(top_k_correct)
    mrr_scores.append(mrr)
    map_scores.append(map_score)
    cosine_sims.append(avg_cosine_sim)

# --- Evaluate test set ---
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
top_k_accuracy = np.mean(top_k_corrects)
mrr = np.mean(mrr_scores)
map_score = np.mean(map_scores)
avg_cosine_similarity = np.mean(cosine_sims)

print(f"\nTest Set Evaluation Metrics:")
print(f"Precision           : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Top-{k} Accuracy    : {top_k_accuracy:.4f}")
print(f"Mean Reciprocal Rank: {mrr:.4f}")
print(f"Mean Average Precision: {map_score:.4f}")
print(f"Average Cosine Similarity: {avg_cosine_similarity:.4f}")

# --- Interactive query section with metrics ---
query_predicted_labels = []
query_true_labels = []
query_top_k_corrects = []
query_mrr_scores = []
query_map_scores = []
query_cosine_sims = []

print("\nEnter an API call sequence and its true label (0 for Benign, 1 for Malware, or 'skip' to classify without label):")
while True:
    query_sequence = input("API Sequence: ").strip()
    if query_sequence.lower() == 'exit':
        break
    if not query_sequence:
        print("Please enter a valid API sequence.")
        continue

    # Prompt for true label
    try:
        true_label = input("True Label (0 for Benign, 1 for Malware, or 'skip' to classify without label): ").strip()
        if true_label.lower() == 'skip':
            true_label = None
        else:
            true_label = int(true_label)
            if true_label not in [0, 1]:
                print("Invalid label. Please enter 0 (Benign), 1 (Malware), or 'skip'.")
                continue
    except ValueError:
        print("Invalid label. Please enter 0 (Benign), 1 (Malware), or 'skip'.")
        continue

    # Classify the sequence
    similarities = [SequenceMatcher(None, query_sequence, seq).ratio() for seq in seq_train]
    closest_idx = np.argmax(similarities)
    query_emb = X_train[closest_idx].reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, k)
    retrieved_labels = [y_train[idx] for idx in I[0]]

    # Generate LLM embedding for query
    query_llm_emb = get_llm_embeddings([query_sequence])[0]
    query_features = np.concatenate([query_llm_emb, [np.mean(retrieved_labels)]])
    predicted_label = classifier.predict([query_features])[0]
    print(f"\nClassification Result: Label: {'Malware' if predicted_label == 1 else 'Benign'}")

    if true_label is not None:
        top_k_correct, mrr, map_score, avg_cosine_sim = compute_additional_metrics(true_label, retrieved_labels, D[0], k)
        query_predicted_labels.append(predicted_label)
        query_true_labels.append(true_label)
        query_top_k_corrects.append(top_k_correct)
        query_mrr_scores.append(mrr)
        query_map_scores.append(map_score)
        query_cosine_sims.append(avg_cosine_sim)

# --- Calculate and display metrics for user queries ---
if query_true_labels:
    query_precision = precision_score(query_true_labels, query_predicted_labels)
    query_recall = recall_score(query_true_labels, query_predicted_labels)
    query_f1 = f1_score(query_true_labels, query_predicted_labels)
    query_accuracy = accuracy_score(query_true_labels, query_predicted_labels)
    query_top_k_accuracy = np.mean(query_top_k_corrects)
    query_mrr = np.mean(query_mrr_scores)
    query_map = np.mean(query_map_scores)  # Fixed typo: query_Map_scores -> query_map_scores
    query_avg_cosine_sim = np.mean(query_cosine_sims)

    print(f"\nUser Query Evaluation Metrics:")
    print(f"Precision           : {query_precision:.4f}")
    print(f"Recall             : {query_recall:.4f}")
    print(f"F1 Score           : {query_f1:.4f}")
    print(f"Accuracy           : {query_accuracy:.4f}")
    print(f"Top-{k} Accuracy    : {query_top_k_accuracy:.4f}")
    print(f"Mean Reciprocal Rank: {query_mrr:.4f}")
    print(f"Mean Average Precision: {query_map:.4f}")
    print(f"Average Cosine Similarity: {query_avg_cosine_sim:.4f}")
else:
    print("\nNo queries with true labels provided for metric calculation.")
