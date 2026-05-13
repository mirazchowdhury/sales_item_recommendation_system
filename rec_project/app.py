import os
import pandas as pd
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = 'data/itemDataset.csv'
EMBEDDING_PKL = 'item_embeddings_cpu.pkl'
OUTPUT_CSV = 'recommendations_output.csv'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


class CPURecSystem:
    def __init__(self):
        self.device = 'cpu'
        print(f"--- Loading Model on {self.device.upper()} ---")
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.df = None
        self.embeddings = None

    def run_pipeline(self):
        if not os.path.exists(INPUT_CSV):
            print(f"Error: {INPUT_CSV} পাওয়া যায়নি!")
            return

        print("Step 1: Reading Data & Processing Serial...")
        self.df = pd.read_csv(INPUT_CSV).dropna(subset=['item_name']).reset_index(drop=True)

        # product_code ক্লিন করা
        self.df['product_code_str'] = self.df['product_code'].astype(str).str.replace("'", "")

        # এমবেডিং জেনারেশন (ব্র্যান্ড ওয়েটিং সহ)
        if os.path.exists(EMBEDDING_PKL):
            with open(EMBEDDING_PKL, 'rb') as f:
                self.embeddings = pickle.load(f)['vecs']
        else:
            print("Generating Embeddings...")
            weighted_names = self.df['item_name'].apply(lambda x: f"{str(x).split()[0]} {x}")
            self.embeddings = self.model.encode(weighted_names.tolist(), show_progress_bar=True, batch_size=32)
            with open(EMBEDDING_PKL, 'wb') as f:
                pickle.dump({'vecs': self.embeddings, 'ids': self.df['item_id'].values}, f)

        print("Step 2: Calculating Recommendations (Product Code + Serial Priority)...")
        final_recommendations = []

        for i in tqdm(range(len(self.df)), desc="Processing"):
            current_item_id = self.df.iloc[i]['item_id']
            current_code = self.df.iloc[i]['product_code_str']

            # ১. একই product_code যাদের আছে তাদের বের করা (সিরিয়াল অনুযায়ী)
            same_code_df = self.df[
                (self.df['product_code_str'] == current_code) & (self.df['item_id'] != current_item_id)]

            # ২. একই কোডের আইটেমগুলো নিয়ে লিস্ট শুরু করা
            rec_list = same_code_df['item_name'].tolist()

            # ৩. যদি ৫টার কম হয়, তবে এমবেডিং দিয়ে বাকিগুলো পূরণ করা
            if len(rec_list) < 5:
                query_vec = self.embeddings[i].reshape(1, -1)
                scores = cosine_similarity(query_vec, self.embeddings).flatten()

                # স্কোর অনুযায়ী সর্ট করা
                related_indices = scores.argsort()[::-1]

                for idx in related_indices:
                    name = self.df.iloc[idx]['item_name']
                    id_val = self.df.iloc[idx]['item_id']
                    # নিজেকে বাদ দেওয়া এবং অলরেডি লিস্টে থাকলে বাদ দেওয়া
                    if id_val != current_item_id and name not in rec_list:
                        rec_list.append(name)
                    if len(rec_list) >= 5:
                        break

            # ৪. প্রথম ৫টি নেওয়া
            final_recommendations.append(" | ".join(rec_list[:5]))

        self.df['top_5_recommendations'] = final_recommendations
        self.df[['item_id', 'item_name', 'product_code', 'top_5_recommendations']].to_csv(OUTPUT_CSV, index=False)
        print(f"Success! Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    CPURecSystem().run_pipeline()