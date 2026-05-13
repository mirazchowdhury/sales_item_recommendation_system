import os
import pandas as pd
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = 'data/itemDataset.csv'
EMBEDDING_PKL = 'item_embeddings_gpu.pkl'
OUTPUT_CSV = 'recommendations_output_gpu.csv'


class GPURecSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self.df = None
        self.embeddings = None

    def run_pipeline(self):
        if not os.path.exists(INPUT_CSV): return

        self.df = pd.read_csv(INPUT_CSV).dropna(subset=['item_name']).reset_index(drop=True)
        self.df['product_code_str'] = self.df['product_code'].astype(str).str.replace("'", "")

        # এমবেডিং তৈরি
        print("Generating Embeddings on GPU...")
        names = self.df['item_name'].tolist()
        self.embeddings = self.model.encode(names, show_progress_bar=True, batch_size=128, convert_to_tensor=True)

        print("Calculating Hybrid Recommendations...")
        final_recs = []

        # দ্রুত খোঁজার জন্য ডিকশনারি ম্যাপিং
        code_map = self.df.groupby('product_code_str').apply(
            lambda x: x[['item_name', 'item_id']].to_dict('records')).to_dict()

        for i in tqdm(range(len(self.df))):
            current_id = self.df.iloc[i]['item_id']
            current_code = self.df.iloc[i]['product_code_str']

            # ১. একই কোড এবং সিরিয়াল অনুযায়ী শুরু
            matches = [item['item_name'] for item in code_map[current_code] if item['item_id'] != current_id]

            # ২. ৫টা না হলে সিমিলারিটি যোগ করা
            if len(matches) < 5:
                query_vec = self.embeddings[i].unsqueeze(0)
                scores = torch.nn.functional.cosine_similarity(query_vec, self.embeddings)
                top_idx = torch.topk(scores, k=min(20, len(self.df))).indices.tolist()

                for idx in top_idx:
                    name = self.df.iloc[idx]['item_name']
                    if name not in matches and self.df.iloc[idx]['item_id'] != current_id:
                        matches.append(name)
                    if len(matches) >= 5: break

            final_recs.append(" | ".join(matches[:5]))

        self.df['top_5_recommendations'] = final_recs
        self.df[['item_id', 'item_name', 'product_code', 'top_5_recommendations']].to_csv(OUTPUT_CSV, index=False)
        print("GPU processing complete!")


if __name__ == "__main__":
    GPURecSystem().run_pipeline()