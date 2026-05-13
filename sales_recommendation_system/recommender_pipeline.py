import pandas as pd
import numpy as np
import os
import torch
import warnings
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from mlxtend.frequent_patterns import fpgrowth, association_rules
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"C:\D drive\sales_recommendation_system"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

INPUT_FILE = os.path.join(DATA_DIR, "merged_sales_history.csv")

MIN_SUPPORT_ITEM = 0.0005
MIN_SUPPORT_SUBCAT = 0.002  # Macro level e support ektu beshi thake
MIN_CONFIDENCE = 0.05
TOP_N = 5

os.makedirs(RESULTS_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==========================================
# UTILITIES
# ==========================================
def encode_units(x):
    return bool(x >= 1)


def parse_frozenset(fs_data):
    if isinstance(fs_data, frozenset):
        return list(fs_data)
    elif isinstance(fs_data, str):
        return list(eval(fs_data))
    return []


# ==========================================
# PIPELINE CLASS
# ==========================================
class AdvancedRecommenderPipeline:
    def __init__(self):
        self.df = None
        self.train_df = None
        self.test_df = None
        self.item_meta = {}

        self.item_rules = pd.DataFrame()
        self.subcat_rules = pd.DataFrame()
        self.occasion_popularity = {}

    def load_and_preprocess_data(self):
        print("\n--- STEP 1: Data Preprocessing & Splitting ---")
        self.df = pd.read_csv(INPUT_FILE)

        # 1. Remove Time from Datetime (Keep only Date)
        self.df['order_date'] = pd.to_datetime(self.df['order_datetime']).dt.date
        self.df = self.df.drop(columns=['order_datetime'], errors='ignore')

        # Save updated raw dataset
        updated_file = os.path.join(DATA_DIR, "merged_sales_history_updated.csv")
        self.df.to_csv(updated_file, index=False)
        print(f"Time removed. Saved updated dataset to {updated_file}")

        # Extract Item Metadata (Category, Sub-Cat, Occasion)
        meta_cols = ['Category', 'Sub_Category', 'Use_Case_Occasion']
        self.item_meta = self.df.set_index('item_name')[meta_cols].drop_duplicates().to_dict('index')

        # Train/Test Split (80/20 based on chronological order_date)
        orders = self.df[['order_id', 'order_date']].drop_duplicates().sort_values('order_date')
        train_orders, test_orders = train_test_split(orders['order_id'], test_size=0.2, shuffle=False)

        self.train_df = self.df[self.df['order_id'].isin(train_orders)].copy()
        self.test_df = self.df[self.df['order_id'].isin(test_orders)].copy()

    def generate_baskets_and_rules(self):
        print("\n--- STEP 2: Building Micro & Macro Baskets (FP-Growth) ---")

        # ----------------------------------------------------
        # MICRO LEVEL: Item Basket & Rules
        # ----------------------------------------------------
        print("Generating Item-Level Basket & Rules...")
        item_basket = (self.train_df.groupby(['order_id', 'item_name'])['quantity']
                       .sum().unstack().reset_index().fillna(0).set_index('order_id'))
        item_basket_bool = item_basket.map(encode_units)
        item_basket_bool.to_csv(os.path.join(DATA_DIR, "basket_item_level.csv"))

        item_freq = fpgrowth(item_basket_bool, min_support=MIN_SUPPORT_ITEM, use_colnames=True)
        if not item_freq.empty:
            self.item_rules = association_rules(item_freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
            self.item_rules.to_csv(os.path.join(DATA_DIR, "rules_item_level.csv"), index=False)

        # ----------------------------------------------------
        # MACRO LEVEL: Sub-Category Basket & Rules
        # ----------------------------------------------------
        print("Generating Sub-Category-Level Basket & Rules...")
        subcat_basket = (self.train_df.groupby(['order_id', 'Sub_Category'])['quantity']
                         .sum().unstack().reset_index().fillna(0).set_index('order_id'))
        subcat_basket_bool = subcat_basket.map(encode_units)
        subcat_basket_bool.to_csv(os.path.join(DATA_DIR, "basket_subcat_level.csv"))

        subcat_freq = fpgrowth(subcat_basket_bool, min_support=MIN_SUPPORT_SUBCAT, use_colnames=True)
        if not subcat_freq.empty:
            self.subcat_rules = association_rules(subcat_freq, metric="confidence", min_threshold=MIN_CONFIDENCE)
            self.subcat_rules.to_csv(os.path.join(DATA_DIR, "rules_subcat_level.csv"), index=False)

        # ----------------------------------------------------
        # OCCASION PROFILING (For Cold Start & Bundling)
        # ----------------------------------------------------
        print("Generating Occasion-Based Popularity (Bundles)...")
        occasion_df = self.train_df.groupby(['Use_Case_Occasion', 'item_name'])['quantity'].sum().reset_index()
        occasion_df = occasion_df.sort_values(by=['Use_Case_Occasion', 'quantity'], ascending=[True, False])
        occasion_df.to_csv(os.path.join(DATA_DIR, "occasion_bundles_and_cold_start.csv"), index=False)

        # Convert to dictionary for fast lookup during recommendation
        for occ in occasion_df['Use_Case_Occasion'].unique():
            top_items = occasion_df[occasion_df['Use_Case_Occasion'] == occ]['item_name'].head(20).tolist()
            self.occasion_popularity[occ] = top_items

    def build_matrices(self):
        print("\n--- STEP 3: Co-occurrence, Collaborative & Content Matrices ---")

        # 1. Co-occurrence Matrix
        item_basket_bool = pd.read_csv(os.path.join(DATA_DIR, "basket_item_level.csv"), index_col=0)
        one_hot_int = item_basket_bool.astype(int)
        co_occ = one_hot_int.T.dot(one_hot_int)
        np.fill_diagonal(co_occ.values, 0)
        co_occ.to_csv(os.path.join(DATA_DIR, "matrix_co_occurrence.csv"))

        # 2. Collaborative Matrix (Item-User)
        item_user = (self.train_df.groupby(['item_name', 'customer_id'])['quantity']
                     .sum().unstack().reset_index().fillna(0).set_index('item_name'))
        item_user.to_csv(os.path.join(DATA_DIR, "matrix_item_user.csv"))

        self.collab_sim = pd.DataFrame(cosine_similarity(item_user), index=item_user.index, columns=item_user.index)
        self.collab_sim.to_csv(os.path.join(DATA_DIR, "similarity_collaborative.csv"))

        # 3. Content Matrix (HuggingFace MiniLM)
        print(f"Generating HuggingFace Embeddings on {device}...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        meta_df = self.train_df[
            ['item_name', 'Category', 'Sub_Category', 'Use_Case_Occasion']].drop_duplicates().reset_index(drop=True)
        meta_df['text'] = meta_df['item_name'] + " " + meta_df['Sub_Category'] + " " + meta_df['Category'] + " " + \
                          meta_df['Use_Case_Occasion']

        embeddings = model.encode(meta_df['text'].tolist(), batch_size=32, show_progress_bar=True, device=device)
        pd.DataFrame(embeddings, index=meta_df['item_name']).to_csv(os.path.join(DATA_DIR, "matrix_embeddings.csv"))

        self.content_sim = pd.DataFrame(cosine_similarity(embeddings), index=meta_df['item_name'],
                                        columns=meta_df['item_name'])
        self.content_sim.to_csv(os.path.join(DATA_DIR, "similarity_content.csv"))

    def hybrid_recommend(self, current_cart):
        """Intelligent Recommendation Engine using Micro, Macro, and Occasion Context"""
        if not current_cart:
            return []

        scores = defaultdict(float)

        # --- Understand the Context ---
        cart_subcats = [self.item_meta[i]['Sub_Category'] for i in current_cart if i in self.item_meta]
        cart_occasions = [self.item_meta[i]['Use_Case_Occasion'] for i in current_cart if i in self.item_meta]

        # Identify Primary Occasion of the Cart (Context Inference)
        primary_occasion = Counter(cart_occasions).most_common(1)[0][0] if cart_occasions else None

        # --- 1. Macro Level (Sub-Category Rules) ---
        target_subcats = set()
        if not self.subcat_rules.empty:
            for subcat in set(cart_subcats):
                matched_macro = self.subcat_rules[
                    self.subcat_rules['antecedents'].apply(lambda x: subcat in parse_frozenset(x))]
                for _, row in matched_macro.iterrows():
                    cons_subcat = parse_frozenset(row['consequents'])[0]
                    target_subcats.add(cons_subcat)

        # --- 2. Micro Level (Item Rules) ---
        if not self.item_rules.empty:
            for item in current_cart:
                matched_micro = self.item_rules[
                    self.item_rules['antecedents'].apply(lambda x: item in parse_frozenset(x))]
                for _, row in matched_micro.iterrows():
                    cons_item = parse_frozenset(row['consequents'])[0]
                    scores[cons_item] += row['lift'] * 3.0  # High weight for direct rule

        # --- 3. Collaborative & Content Similarity ---
        for item in current_cart:
            if item in self.collab_sim.index:
                for sim_item, s_score in self.collab_sim[item].nlargest(20).items():
                    scores[sim_item] += s_score * 1.0
            if item in self.content_sim.index:
                for sim_item, s_score in self.content_sim[item].nlargest(20).items():
                    scores[sim_item] += s_score * 0.5

        # --- 4. Contextual Filtering & Macro Boosting ---
        final_scores = {}
        for item, score in scores.items():
            if item in current_cart or item not in self.item_meta:
                continue

            meta = self.item_meta[item]

            # Diversity Rule: Do not recommend identical sub-categories
            if meta['Sub_Category'] in cart_subcats:
                continue

            # Macro Boost: If this item's sub-category is predicted by Macro Rules
            if meta['Sub_Category'] in target_subcats:
                score *= 2.0

                # Contextual Filter: Does it fit the Primary Occasion?
            if primary_occasion and meta['Use_Case_Occasion'] == primary_occasion:
                score *= 1.5  # Boost if context matches perfectly
            elif meta['Category'] == 'Grocery':  # Allow universal staples/spices
                score *= 1.0
            else:
                score *= 0.5  # Penalize if out of context (e.g. Toilet Cleaner while buying Biryani items)

            final_scores[item] = score

        # --- 5. Cold Start / Occasion Bundling Fallback ---
        sorted_recs = [item[0] for item in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)]

        # If we couldn't find enough items, fill with the popular items for this specific occasion!
        if len(sorted_recs) < TOP_N and primary_occasion:
            occasion_bundle = self.occasion_popularity.get(primary_occasion, [])
            for bundle_item in occasion_bundle:
                if len(sorted_recs) >= TOP_N:
                    break
                if bundle_item not in current_cart and bundle_item not in sorted_recs:
                    if self.item_meta.get(bundle_item, {}).get('Sub_Category') not in cart_subcats:
                        sorted_recs.append(bundle_item)

        return sorted_recs[:TOP_N]

    def evaluate_and_save_results(self):
        print("\n--- STEP 4: Evaluation Metrics ---")

        test_orders = self.test_df.groupby('order_id')['item_name'].apply(list).to_dict()
        eval_records = []

        for order_id, items in tqdm(test_orders.items()):
            if len(items) < 2: continue

            split_idx = max(1, len(items) // 2)
            cart = items[:split_idx]
            actual = items[split_idx:]

            predictions = self.hybrid_recommend(cart)

            hit = int(any(p in actual for p in predictions))
            precision = sum(1 for p in predictions if p in actual) / len(predictions) if predictions else 0
            recall = sum(1 for p in predictions if p in actual) / len(actual) if actual else 0

            eval_records.append({'order_id': order_id, 'hit': hit, 'precision': precision, 'recall': recall})

        eval_df = pd.DataFrame(eval_records)
        metrics = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Test_Orders': len(eval_df),
            'Hit_Rate': eval_df['hit'].mean(),
            'Mean_Precision': eval_df['precision'].mean(),
            'Mean_Recall': eval_df['recall'].mean()
        }

        print("\n=== EVALUATION RESULTS ===")
        for k, v in metrics.items(): print(f"{k}: {v}")

        metrics_file = os.path.join(RESULTS_DIR, f"eval_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        print(f"\nPipeline completed! All features, matrices, and results saved in: {DATA_DIR}")


if __name__ == "__main__":
    pipeline = AdvancedRecommenderPipeline()
    pipeline.load_and_preprocess_data()
    pipeline.generate_baskets_and_rules()
    pipeline.build_matrices()
    pipeline.evaluate_and_save_results()