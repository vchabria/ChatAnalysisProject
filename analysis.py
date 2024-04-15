import csv
import pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def clean_text(text, words_to_remove):
    querywords = text.split()
    resultwords  = [word for word in querywords if word.lower() not in words_to_remove]
    result = ' '.join(resultwords)
    return result

def sentiment_analysis(text, words_to_remove):
    cleaned_text = clean_text(text, words_to_remove)
    inputs = tokenizer(cleaned_text, return_tensors="pt")
    outputs = model(**inputs)
    scores = torch.softmax(outputs[0][0], dim=-1).detach().numpy()
    return np.argmax(scores)

def process_csv(csv_file, output_file, words_to_remove):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        with open(output_file, 'w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(header + ['Sentiment Label'])

            for row in reader:
                text = row[1] 
                sentiment_label = sentiment_analysis(text, words_to_remove)
                row.append(sentiment_label)
                writer.writerow(row)

def plot_sentiment_analysis(csv_file):
    df = pd.read_csv(csv_file)
    df['Message Date'] = pd.to_datetime(df['Message Date'])
    incoming_df = df[df['Type'] == 'Incoming']
    outgoing_df = df[df['Type'] == 'Outgoing']
    incoming_grouped = incoming_df.groupby(incoming_df['Message Date'].dt.date)['Sentiment Label'].value_counts()
    outgoing_grouped = outgoing_df.groupby(outgoing_df['Message Date'].dt.date)['Sentiment Label'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.plot(incoming_grouped.index, incoming_grouped.values, color='red', label='Incoming')
    plt.plot(outgoing_grouped.index, outgoing_grouped.values, color='blue', label='Outgoing')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Count')
    plt.title('Sentiment Count Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_file = "vansh-text.csv"
    processed_file = "processed_output.csv"
    words_to_remove = [
    'the', 'is', 'and', 'are', 'to', 'a', 'in', 'it', 'you', 'of', 'for', 'on', 'that', 'this', 'with', 
    'have', 'be', 'as', 'at', 'your', 'was', 'we', 'can', 'my', 'or', 'if', 'but', 'from', 'they', 'will', 
    'an', 'what', 'there', 'so', 'me', 'all', 'one', 'by', 'like', 'her', 'has', 'which', 'out', 'up', 
    'who', 'do', 'their', 'not', 'him', 'his', 'lol', 'omg', 'idk', 'btw', 'fyi', 'tbh', 'np', 'pls', 
    'plz', 'thx', 'ty', 'u', 'ur', 'k', 'ok', 'yeah', 'nope', 'yep'
]
    process_csv(input_file, processed_file, words_to_remove)
    plot_sentiment_analysis(processed_file)