import os
import ast
import json
import pickle
import re
import warnings
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


# =========================================================
# Utility Functions
# =========================================================

def safe_list(value):
    if isinstance(value, list):
        return value

    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, tuple) or isinstance(parsed, set):
            return list(parsed)

        return [parsed]

    except Exception:
        return [x.strip() for x in text.split(",") if x.strip()]


def unique_keep_order(values):
    seen = set()
    output = []

    for value in values:
        value = str(value).strip()

        if value and value not in seen:
            output.append(value)
            seen.add(value)

    return output


def top_counter_items(counter_obj, limit=50):
    if not counter_obj:
        return []

    return counter_obj.most_common(limit)


def normalize_text_key(value):
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# =========================================================
# Dynamic Product Grouping
# =========================================================

PRODUCT_STOPWORDS = {
    "pure", "fresh", "special", "regular", "premium", "super", "new",
    "best", "classic", "original", "natural", "extra", "large", "small",
    "medium", "big", "mini", "family", "value", "combo", "free",
    "buy", "get", "pack", "packet", "bottle", "poly", "pet", "bag",
    "box", "tin", "can", "jar", "pouch", "refill", "with", "without",
    "and", "the", "for", "pcs", "piece", "pieces", "pads", "roll",
    "gram", "grams", "powder", "full", "kg", "gm", "g", "ml", "ltr",
    "lt", "liter", "litre"
}


def normalize_name_for_group(name):
    text = str(name).lower()

    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[-_/]", " ", text)

    text = re.sub(
        r"\b\d+(\.\d+)?\s*(kg|g|gm|gram|grams|l|lt|ltr|liter|litre|ml|pcs|pc|piece|pieces|pads|roll|mm|cm|inch)\b",
        " ",
        text
    )

    text = re.sub(r"\b\d+(\.\d+)?\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_product_name(name):
    text = normalize_name_for_group(name)
    tokens = []

    for token in text.split():
        token = token.strip()

        if len(token) <= 2:
            continue

        if token in PRODUCT_STOPWORDS:
            continue

        tokens.append(token)

    return tokens


def build_product_group_map(meta_df):
    subcat_token_counter = defaultdict(Counter)
    subcat_item_count = defaultdict(int)

    for _, row in meta_df.iterrows():
        subcat = str(row["Sub_Category"])
        tokens = tokenize_product_name(row["item_name"])

        subcat_item_count[subcat] += 1

        for token in set(tokens):
            subcat_token_counter[subcat][token] += 1

    product_group_map = {}
    product_group_label = {}

    for _, row in meta_df.iterrows():
        item_id = str(row["item_id"])
        item_name = str(row["item_name"])
        subcat = str(row["Sub_Category"])

        tokens = tokenize_product_name(item_name)

        if not tokens:
            group_key = f"{subcat}__unknown"
            product_group_map[item_id] = group_key
            product_group_label[item_id] = subcat
            continue

        scored_tokens = []

        for token in tokens:
            token_count = subcat_token_counter[subcat][token]
            total_items = max(subcat_item_count[subcat], 1)
            token_ratio = token_count / total_items

            if token_ratio > 0.70:
                continue

            score = token_count * 10 + len(token)
            scored_tokens.append((score, token))

        if scored_tokens:
            scored_tokens = sorted(scored_tokens, reverse=True)
            anchor_token = scored_tokens[0][1]
        else:
            anchor_token = subcat.lower().replace(" ", "_")

        group_key = f"{subcat}__{anchor_token}"

        product_group_map[item_id] = group_key
        product_group_label[item_id] = anchor_token

    return product_group_map, product_group_label


def build_group_best_items(main_df, product_group_map):
    temp_df = main_df.copy()
    temp_df["item_id"] = temp_df["item_id"].astype(str)
    temp_df["product_group"] = temp_df["item_id"].map(product_group_map)

    temp_df = temp_df.dropna(subset=["product_group"])

    occasion_group_best_item = defaultdict(dict)
    global_group_best_item = {}

    occasion_subcat_best_item = defaultdict(dict)
    global_subcat_best_item = {}

    occasion_group_rank = (
        temp_df
        .groupby(["Use_Case_Occasion", "product_group", "item_id"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(
            ["Use_Case_Occasion", "product_group", "quantity"],
            ascending=[True, True, False]
        )
    )

    for (occasion, group), group_df in occasion_group_rank.groupby(["Use_Case_Occasion", "product_group"]):
        best_item = str(group_df.iloc[0]["item_id"])
        occasion_group_best_item[occasion][group] = best_item

    global_group_rank = (
        temp_df
        .groupby(["product_group", "item_id"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(["product_group", "quantity"], ascending=[True, False])
    )

    for group, group_df in global_group_rank.groupby("product_group"):
        best_item = str(group_df.iloc[0]["item_id"])
        global_group_best_item[group] = best_item

    occasion_subcat_rank = (
        temp_df
        .groupby(["Use_Case_Occasion", "Sub_Category", "item_id"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(
            ["Use_Case_Occasion", "Sub_Category", "quantity"],
            ascending=[True, True, False]
        )
    )

    for (occasion, subcat), group_df in occasion_subcat_rank.groupby(["Use_Case_Occasion", "Sub_Category"]):
        best_item = str(group_df.iloc[0]["item_id"])
        occasion_subcat_best_item[occasion][subcat] = best_item

    global_subcat_rank = (
        temp_df
        .groupby(["Sub_Category", "item_id"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(["Sub_Category", "quantity"], ascending=[True, False])
    )

    for subcat, group_df in global_subcat_rank.groupby("Sub_Category"):
        best_item = str(group_df.iloc[0]["item_id"])
        global_subcat_best_item[subcat] = best_item

    return (
        dict(occasion_group_best_item),
        global_group_best_item,
        dict(occasion_subcat_best_item),
        global_subcat_best_item
    )


# =========================================================
# Recommendation Model
# =========================================================

class HybridRecommenderModel:
    def __init__(self):
        self.item_rules = pd.DataFrame()
        self.subcat_rules = pd.DataFrame()

        self.collab_sim = None
        self.content_sim = None

        self.occasion_popularity = {}
        self.item_meta = {}
        self.id_to_name = {}
        self.name_to_id = {}
        self.normalized_name_to_id = {}

        self.popularity_list = []
        self.item_pair_counter = {}
        self.subcat_pair_counter = {}

        self.product_group_map = {}
        self.product_group_label = {}
        self.occasion_group_best_item = {}
        self.global_group_best_item = {}
        self.occasion_subcat_best_item = {}
        self.global_subcat_best_item = {}

        self.training_report = {}
        self.last_trained = None

    def resolve_item_id(self, item_value):
        item_value = str(item_value).strip()

        if item_value in self.item_meta:
            return item_value

        if item_value in self.name_to_id:
            return self.name_to_id[item_value]

        normalized_name = normalize_text_key(item_value)

        if normalized_name in self.normalized_name_to_id:
            return self.normalized_name_to_id[normalized_name]

        return None

    def get_item_name(self, item_id):
        return self.id_to_name.get(str(item_id), str(item_id))

    def get_product_group(self, item_id):
        item_id = str(item_id)
        return self.product_group_map.get(item_id, item_id)

    def get_product_group_label(self, item_id):
        item_id = str(item_id)
        return self.product_group_label.get(item_id, self.get_product_group(item_id))

    def normalize_score(self, score):
        score = max(float(score), 0.0)
        score = np.log1p(score) * 3.0
        return round(min(score, 20.0), 4)

    def apply_dataset_preference_boost(self, item_id, score, primary_occasion):
        item_id = str(item_id)
        meta = self.item_meta.get(item_id, {})

        group_key = self.get_product_group(item_id)
        subcat = meta.get("Sub_Category")

        new_score = float(score)

        if primary_occasion:
            occasion_best_group_item = self.occasion_group_best_item.get(primary_occasion, {}).get(group_key)

            if occasion_best_group_item == item_id:
                new_score *= 1.35

            occasion_best_subcat_item = self.occasion_subcat_best_item.get(primary_occasion, {}).get(subcat)

            if occasion_best_subcat_item == item_id:
                new_score *= 1.20

        global_best_group_item = self.global_group_best_item.get(group_key)

        if global_best_group_item == item_id:
            new_score *= 1.15

        global_best_subcat_item = self.global_subcat_best_item.get(subcat)

        if global_best_subcat_item == item_id:
            new_score *= 1.10

        return new_score

    def diversity_rerank(self, candidate_items, final_scores, resolved_cart, top_n):
        selected_items = []
        selected_groups = set()
        selected_subcats = set()

        cart_groups = {
            self.get_product_group(item_id)
            for item_id in resolved_cart
        }

        cart_subcats = {
            self.item_meta.get(item_id, {}).get("Sub_Category", "")
            for item_id in resolved_cart
        }

        for item_id in candidate_items:
            if item_id in resolved_cart:
                continue

            if item_id not in self.item_meta:
                continue

            group_key = self.get_product_group(item_id)
            subcat = self.item_meta.get(item_id, {}).get("Sub_Category", "")

            if group_key in cart_groups:
                continue

            if subcat in cart_subcats:
                continue

            if group_key in selected_groups:
                continue

            if subcat in selected_subcats:
                continue

            selected_items.append(item_id)
            selected_groups.add(group_key)
            selected_subcats.add(subcat)

            if len(selected_items) >= top_n:
                break

        if len(selected_items) < top_n:
            selected_subcat_count = Counter()

            for selected_item in selected_items:
                selected_subcat = self.item_meta.get(selected_item, {}).get("Sub_Category", "")
                selected_subcat_count[selected_subcat] += 1

            for item_id in candidate_items:
                if len(selected_items) >= top_n:
                    break

                if item_id in resolved_cart:
                    continue

                if item_id in selected_items:
                    continue

                if item_id not in self.item_meta:
                    continue

                group_key = self.get_product_group(item_id)
                subcat = self.item_meta.get(item_id, {}).get("Sub_Category", "")

                if group_key in cart_groups:
                    continue

                if subcat in cart_subcats:
                    continue

                if group_key in selected_groups:
                    continue

                if selected_subcat_count[subcat] >= 2:
                    continue

                selected_items.append(item_id)
                selected_groups.add(group_key)
                selected_subcat_count[subcat] += 1

        if len(selected_items) < top_n:
            selected_subcat_count = Counter()

            for selected_item in selected_items:
                selected_subcat = self.item_meta.get(selected_item, {}).get("Sub_Category", "")
                selected_subcat_count[selected_subcat] += 1

            for item_id in self.popularity_list:
                if len(selected_items) >= top_n:
                    break

                if item_id in resolved_cart:
                    continue

                if item_id in selected_items:
                    continue

                if item_id not in self.item_meta:
                    continue

                group_key = self.get_product_group(item_id)
                subcat = self.item_meta.get(item_id, {}).get("Sub_Category", "")

                if group_key in cart_groups:
                    continue

                if subcat in cart_subcats:
                    continue

                if group_key in selected_groups:
                    continue

                if selected_subcat_count[subcat] >= 2:
                    continue

                selected_items.append(item_id)
                selected_groups.add(group_key)
                selected_subcat_count[subcat] += 1

                final_scores[item_id] = max(final_scores.get(item_id, 0), 0.5)

        return selected_items

    def recommend(self, current_cart, top_n=5, customer_id=None):
        debug_info = {
            "input_item_ids": [],
            "input_item_names": [],
            "missing_items": [],
            "detected_occasion": None,
            "blocked_subcategories": [],
            "cart_product_groups": [],
            "target_subcategories": [],
            "signals_used": [],
            "raw_scores": {},
            "score_breakdown": {},
            "product_groups": {},
            "reasoning": []
        }

        resolved_cart = []

        for item in current_cart:
            item_id = self.resolve_item_id(item)

            if item_id:
                resolved_cart.append(item_id)
            else:
                debug_info["missing_items"].append(str(item))

        resolved_cart = unique_keep_order(resolved_cart)

        if not resolved_cart:
            debug_info["reasoning"].append("No cart item matched item id or item name in the trained catalog.")
            return [], debug_info

        debug_info["input_item_ids"] = resolved_cart
        debug_info["input_item_names"] = [self.get_item_name(item_id) for item_id in resolved_cart]

        scores = defaultdict(float)
        signal_breakdown = defaultdict(lambda: defaultdict(float))

        cart_subcats = []
        cart_occasions = []
        cart_groups = []

        for item_id in resolved_cart:
            meta = self.item_meta.get(item_id, {})
            subcat = meta.get("Sub_Category")
            occasion = meta.get("Use_Case_Occasion")

            if subcat:
                cart_subcats.append(subcat)

            if occasion:
                cart_occasions.append(occasion)

            cart_groups.append(self.get_product_group_label(item_id))

        cart_subcats = unique_keep_order(cart_subcats)
        cart_groups = unique_keep_order(cart_groups)

        if cart_occasions:
            primary_occasion = Counter(cart_occasions).most_common(1)[0][0]
        else:
            primary_occasion = None

        debug_info["detected_occasion"] = primary_occasion
        debug_info["blocked_subcategories"] = cart_subcats
        debug_info["cart_product_groups"] = cart_groups

        target_subcats = set()

        for subcat in cart_subcats:
            related_subcats = self.subcat_pair_counter.get(subcat, Counter())

            for next_subcat, pair_count in top_counter_items(related_subcats, limit=12):
                if next_subcat not in cart_subcats:
                    target_subcats.add(next_subcat)

        if target_subcats:
            debug_info["signals_used"].append("subcategory_pair_counter")

        if isinstance(self.subcat_rules, pd.DataFrame) and not self.subcat_rules.empty:
            for subcat in cart_subcats:
                matched_rules = self.subcat_rules[
                    self.subcat_rules["antecedents"].apply(lambda x: subcat in x)
                ]

                for _, row in matched_rules.iterrows():
                    for cons_subcat in row["consequents"]:
                        if cons_subcat not in cart_subcats:
                            target_subcats.add(cons_subcat)

            debug_info["signals_used"].append("subcategory_association_rules")

        debug_info["target_subcategories"] = sorted(target_subcats)

        if isinstance(self.item_rules, pd.DataFrame) and not self.item_rules.empty:
            for item_id in resolved_cart:
                matched_rules = self.item_rules[
                    self.item_rules["antecedents"].apply(lambda x: item_id in x)
                ]

                for _, row in matched_rules.iterrows():
                    rule_score = float(row.get("lift", 1.0)) * 2.0
                    rule_score += float(row.get("confidence", 0.0)) * 3.0
                    rule_score += float(row.get("support", 0.0)) * 10.0

                    for cons_item in row["consequents"]:
                        scores[cons_item] += rule_score
                        signal_breakdown[cons_item]["item_rule"] += rule_score

            debug_info["signals_used"].append("item_association_rules")

        for item_id in resolved_cart:
            related_items = self.item_pair_counter.get(item_id, Counter())

            for pair_item, pair_count in top_counter_items(related_items, limit=100):
                pair_score = np.log1p(pair_count) * 1.8
                scores[pair_item] += pair_score
                signal_breakdown[pair_item]["item_pair_counter"] += pair_score

        if self.item_pair_counter:
            debug_info["signals_used"].append("item_pair_counter")

        if self.collab_sim is not None:
            for item_id in resolved_cart:
                if item_id in self.collab_sim.index:
                    similar_items = self.collab_sim.loc[item_id].drop(labels=resolved_cart, errors="ignore")

                    for sim_item, sim_score in similar_items.nlargest(80).items():
                        add_score = float(sim_score) * 1.0
                        scores[sim_item] += add_score
                        signal_breakdown[sim_item]["collaborative_similarity"] += add_score

            debug_info["signals_used"].append("collaborative_similarity")

        if self.content_sim is not None:
            for item_id in resolved_cart:
                if item_id in self.content_sim.index:
                    similar_items = self.content_sim.loc[item_id].drop(labels=resolved_cart, errors="ignore")

                    for sim_item, sim_score in similar_items.nlargest(80).items():
                        add_score = float(sim_score) * 0.25
                        scores[sim_item] += add_score
                        signal_breakdown[sim_item]["content_similarity"] += add_score

            debug_info["signals_used"].append("content_similarity")

        final_scores = {}

        cart_product_groups = {
            self.get_product_group(item_id)
            for item_id in resolved_cart
        }

        for item_id, base_score in scores.items():
            if item_id in resolved_cart:
                continue

            if item_id not in self.item_meta:
                continue

            meta = self.item_meta[item_id]
            item_subcat = meta.get("Sub_Category")
            item_category = meta.get("Category")
            item_occasion = meta.get("Use_Case_Occasion")
            item_group = self.get_product_group(item_id)

            if item_subcat in cart_subcats:
                continue

            if item_group in cart_product_groups:
                continue

            score = float(base_score)

            if item_subcat in target_subcats:
                score *= 2.0
                signal_breakdown[item_id]["target_subcategory_boost"] += score

            if primary_occasion and item_occasion == primary_occasion:
                score *= 1.8
                signal_breakdown[item_id]["occasion_boost"] += score

            elif item_category == "Grocery":
                score *= 1.20
                signal_breakdown[item_id]["grocery_context_boost"] += score

            else:
                score *= 0.65

            score = self.apply_dataset_preference_boost(
                item_id=item_id,
                score=score,
                primary_occasion=primary_occasion
            )

            final_scores[item_id] = self.normalize_score(score)

        candidate_items = [
            item_id
            for item_id, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        if len(candidate_items) < top_n and target_subcats:
            debug_info["reasoning"].append("Adding popular candidates from target subcategories.")

            target_items = (
                pd.DataFrame.from_dict(self.item_meta, orient="index")
                .reset_index()
                .rename(columns={"index": "item_id"})
            )

            target_item_ids = set(
                target_items[target_items["Sub_Category"].isin(target_subcats)]["item_id"].tolist()
            )

            for pop_item in self.popularity_list:
                if pop_item in resolved_cart:
                    continue

                if pop_item in candidate_items:
                    continue

                if pop_item in target_item_ids:
                    candidate_items.append(pop_item)
                    final_scores[pop_item] = max(final_scores.get(pop_item, 0), 1.25)

                if len(candidate_items) >= top_n * 4:
                    break

        if primary_occasion:
            debug_info["reasoning"].append("Adding occasion based candidates.")

            for bundle_item in self.occasion_popularity.get(primary_occasion, []):
                if bundle_item in resolved_cart:
                    continue

                if bundle_item not in candidate_items:
                    candidate_items.append(bundle_item)

                final_scores[bundle_item] = max(final_scores.get(bundle_item, 0), 1.0)

                if len(candidate_items) >= top_n * 5:
                    break

        if len(candidate_items) < top_n:
            debug_info["reasoning"].append("Adding global popular candidates.")

            for pop_item in self.popularity_list:
                if pop_item in resolved_cart:
                    continue

                if pop_item in candidate_items:
                    continue

                candidate_items.append(pop_item)
                final_scores[pop_item] = max(final_scores.get(pop_item, 0), 0.5)

                if len(candidate_items) >= top_n * 5:
                    break

        debug_info["reasoning"].append("Dataset based product group diversity applied.")

        top_recs = self.diversity_rerank(
            candidate_items=candidate_items,
            final_scores=final_scores,
            resolved_cart=resolved_cart,
            top_n=top_n
        )

        debug_info["raw_scores"] = {
            self.get_item_name(item_id): round(float(final_scores.get(item_id, 0)), 4)
            for item_id in top_recs
        }

        debug_info["score_breakdown"] = {
            self.get_item_name(item_id): {
                signal_name: round(float(signal_value), 4)
                for signal_name, signal_value in signal_breakdown[item_id].items()
            }
            for item_id in top_recs
        }

        debug_info["product_groups"] = {
            self.get_item_name(item_id): self.get_product_group_label(item_id)
            for item_id in top_recs
        }

        return top_recs, debug_info


# =========================================================
# Training Pipeline
# =========================================================

def merge_new_json_data(data_dir):
    json_path = os.path.join(data_dir, "new_sales.json")
    csv_path = os.path.join(data_dir, "merged_sales_history_updated.csv")

    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    old_df = pd.read_csv(csv_path)
    new_df = pd.DataFrame(new_data)

    merged_df = pd.concat([old_df, new_df], ignore_index=True)
    merged_df.to_csv(csv_path, index=False)

    archive_name = f"processed_sales_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    os.rename(json_path, os.path.join(data_dir, archive_name))

    print(f"New sales data merged. Archived as {archive_name}")


def build_pair_counter(transactions):
    pair_counter = defaultdict(Counter)

    for basket in transactions:
        basket = unique_keep_order(basket)

        if len(basket) < 2:
            continue

        for left_item in basket:
            for right_item in basket:
                if left_item != right_item:
                    pair_counter[left_item][right_item] += 1

    return dict(pair_counter)


def build_association_rules(transactions, min_support, min_confidence, label):
    try:
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        from mlxtend.preprocessing import TransactionEncoder

    except Exception as exc:
        print(f"{label} rules skipped because mlxtend is not installed. Reason: {exc}")
        return pd.DataFrame()

    valid_transactions = []

    for basket in transactions:
        clean_basket = unique_keep_order(basket)

        if len(clean_basket) >= 2:
            valid_transactions.append(clean_basket)

    if not valid_transactions:
        print(f"{label} rules skipped because no valid baskets were found.")
        return pd.DataFrame()

    try:
        te = TransactionEncoder()
        encoded = te.fit(valid_transactions).transform(valid_transactions, sparse=True)
        basket_df = pd.DataFrame.sparse.from_spmatrix(encoded, columns=te.columns_)

        frequent_itemsets = fpgrowth(
            basket_df,
            min_support=min_support,
            use_colnames=True
        )

        if frequent_itemsets.empty:
            print(f"{label} frequent itemsets empty at min support {min_support}.")
            return pd.DataFrame()

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        if rules.empty:
            print(f"{label} rules empty at min confidence {min_confidence}.")
            return pd.DataFrame()

        rules = rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=False
        ).reset_index(drop=True)

        print(f"{label} baskets read: {len(valid_transactions)}")
        print(f"{label} frequent itemsets: {len(frequent_itemsets)}")
        print(f"{label} association rules: {len(rules)}")

        return rules

    except Exception as exc:
        print(f"{label} rules failed. Reason: {exc}")
        return pd.DataFrame()


def load_item_baskets(data_dir, name_to_id, normalized_name_to_id):
    basket_path = os.path.join(data_dir, "basket_item_level_list.csv")
    basket_df = pd.read_csv(basket_path)

    if "items" not in basket_df.columns:
        raise ValueError("basket_item_level_list.csv must contain an items column.")

    item_baskets = []
    missing_names = Counter()

    for value in basket_df["items"].dropna():
        names = safe_list(value)
        ids = []

        for item_name in names:
            item_name = str(item_name).strip()
            item_id = name_to_id.get(item_name)

            if not item_id:
                item_id = normalized_name_to_id.get(normalize_text_key(item_name))

            if item_id:
                ids.append(item_id)
            else:
                missing_names[item_name] += 1

        ids = unique_keep_order(ids)

        if len(ids) >= 2:
            item_baskets.append(ids)

    print(f"Item basket file rows: {len(basket_df)}")
    print(f"Valid item baskets after name to id mapping: {len(item_baskets)}")
    print(f"Missing item names from basket mapping: {len(missing_names)}")

    if missing_names:
        print("Top missing basket item names:")
        for name, count in missing_names.most_common(10):
            print(f"{name}: {count}")

    return item_baskets


def load_subcat_baskets(data_dir):
    basket_path = os.path.join(data_dir, "basket_subcat_level_list.csv")
    basket_df = pd.read_csv(basket_path)

    if "sub_categories" not in basket_df.columns:
        raise ValueError("basket_subcat_level_list.csv must contain a sub categories column.")

    subcat_baskets = []

    for value in basket_df["sub_categories"].dropna():
        subcats = unique_keep_order(safe_list(value))

        if len(subcats) >= 2:
            subcat_baskets.append(subcats)

    print(f"Subcategory basket file rows: {len(basket_df)}")
    print(f"Valid subcategory baskets: {len(subcat_baskets)}")

    return subcat_baskets


def build_content_similarity(meta_df):
    text_values = (
        meta_df["item_name"].fillna("").astype(str)
        + " "
        + meta_df["Category"].fillna("").astype(str)
        + " "
        + meta_df["Sub_Category"].fillna("").astype(str)
        + " "
        + meta_df["Use_Case_Occasion"].fillna("").astype(str)
    ).tolist()

    item_ids = meta_df["item_id"].astype(str).tolist()

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer on {device}")

        st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        embeddings = st_model.encode(
            text_values,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        sim = cosine_similarity(embeddings)
        print("Content similarity built with SentenceTransformer.")

    except Exception as exc:
        print(f"SentenceTransformer unavailable. Using TFIDF fallback. Reason: {exc}")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=3000
        )

        vectors = vectorizer.fit_transform(text_values)
        sim = cosine_similarity(vectors)
        print("Content similarity built with TFIDF fallback.")

    return pd.DataFrame(sim, index=item_ids, columns=item_ids)


def train_and_save_model(data_dir):
    merge_new_json_data(data_dir)

    print("Starting model training.")
    print(f"Data directory: {data_dir}")

    model = HybridRecommenderModel()

    main_path = os.path.join(data_dir, "merged_sales_history_updated.csv")
    main_df = pd.read_csv(main_path)

    required_cols = [
        "customer_id",
        "item_id",
        "item_name",
        "Category",
        "Sub_Category",
        "Use_Case_Occasion",
        "quantity"
    ]

    missing_cols = [col for col in required_cols if col not in main_df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in merged sales file: {missing_cols}")

    main_df["item_id"] = main_df["item_id"].astype(str).str.strip()
    main_df["item_name"] = main_df["item_name"].astype(str).str.strip()
    main_df["customer_id"] = main_df["customer_id"].astype(str).str.strip()
    main_df["Category"] = main_df["Category"].astype(str).str.strip()
    main_df["Sub_Category"] = main_df["Sub_Category"].astype(str).str.strip()
    main_df["Use_Case_Occasion"] = main_df["Use_Case_Occasion"].astype(str).str.strip()
    main_df["quantity"] = pd.to_numeric(main_df["quantity"], errors="coerce").fillna(1)

    meta_df = (
        main_df[["item_id", "item_name", "Category", "Sub_Category", "Use_Case_Occasion"]]
        .drop_duplicates(subset=["item_id"])
        .reset_index(drop=True)
    )

    model.id_to_name = dict(zip(meta_df["item_id"], meta_df["item_name"]))
    model.name_to_id = dict(zip(meta_df["item_name"], meta_df["item_id"]))
    model.normalized_name_to_id = {
        normalize_text_key(name): item_id
        for name, item_id in model.name_to_id.items()
    }

    model.item_meta = (
        meta_df
        .set_index("item_id")[["item_name", "Category", "Sub_Category", "Use_Case_Occasion"]]
        .to_dict("index")
    )

    model.product_group_map, model.product_group_label = build_product_group_map(meta_df)

    (
        model.occasion_group_best_item,
        model.global_group_best_item,
        model.occasion_subcat_best_item,
        model.global_subcat_best_item
    ) = build_group_best_items(
        main_df=main_df,
        product_group_map=model.product_group_map
    )

    print(f"Product groups created: {len(set(model.product_group_map.values()))}")

    model.popularity_list = (
        main_df.groupby("item_id")["quantity"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    print(f"Sales rows: {len(main_df)}")

    if "order_id" in main_df.columns:
        print(f"Unique orders: {main_df['order_id'].nunique()}")
    else:
        print("Unique orders: order id column missing")

    print(f"Unique items: {main_df['item_id'].nunique()}")
    print(f"Unique customers: {main_df['customer_id'].nunique()}")

    item_baskets = load_item_baskets(
        data_dir=data_dir,
        name_to_id=model.name_to_id,
        normalized_name_to_id=model.normalized_name_to_id
    )

    subcat_baskets = load_subcat_baskets(data_dir)

    model.item_pair_counter = build_pair_counter(item_baskets)
    model.subcat_pair_counter = build_pair_counter(subcat_baskets)

    print(f"Item pair counter keys: {len(model.item_pair_counter)}")
    print(f"Subcategory pair counter keys: {len(model.subcat_pair_counter)}")

    model.item_rules = build_association_rules(
        transactions=item_baskets,
        min_support=0.0002,
        min_confidence=0.02,
        label="Item"
    )

    model.subcat_rules = build_association_rules(
        transactions=subcat_baskets,
        min_support=0.001,
        min_confidence=0.03,
        label="Subcategory"
    )

    occasion_path = os.path.join(data_dir, "occasion_profile_basket.csv")

    if os.path.exists(occasion_path):
        occ_basket_df = pd.read_csv(occasion_path)

        for _, row in occ_basket_df.iterrows():
            occ_name = row.get("Use_Case_Occasion")

            if pd.isna(occ_name):
                continue

            occ_name = str(occ_name).strip()
            top_subcats = safe_list(row.get("top_3_sub_categories", []))

            occ_items = (
                main_df[
                    (main_df["Use_Case_Occasion"] == occ_name)
                    & (main_df["Sub_Category"].isin(top_subcats))
                ]
                .groupby("item_id")["quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(50)
                .index
                .tolist()
            )

            model.occasion_popularity[occ_name] = occ_items

        print(f"Occasion profiles loaded: {len(model.occasion_popularity)}")

    else:
        print("occasion_profile_basket.csv not found.")

    item_user = (
        main_df.groupby(["item_id", "customer_id"])["quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    if len(item_user) >= 2:
        collab_matrix = cosine_similarity(item_user)
        model.collab_sim = pd.DataFrame(
            collab_matrix,
            index=item_user.index,
            columns=item_user.index
        )

        print(f"Collaborative similarity matrix shape: {model.collab_sim.shape}")

    else:
        print("Collaborative similarity skipped because too few items were found.")

    model.content_sim = build_content_similarity(meta_df)
    print(f"Content similarity matrix shape: {model.content_sim.shape}")

    model.last_trained = pd.Timestamp.now()

    if "order_id" in main_df.columns:
        unique_orders = int(main_df["order_id"].nunique())
    else:
        unique_orders = None

    model.training_report = {
        "last_trained": str(model.last_trained),
        "sales_rows": int(len(main_df)),
        "unique_orders": unique_orders,
        "unique_items": int(main_df["item_id"].nunique()),
        "unique_customers": int(main_df["customer_id"].nunique()),
        "item_baskets": int(len(item_baskets)),
        "subcat_baskets": int(len(subcat_baskets)),
        "item_pair_counter_keys": int(len(model.item_pair_counter)),
        "subcat_pair_counter_keys": int(len(model.subcat_pair_counter)),
        "item_rules": int(len(model.item_rules)) if isinstance(model.item_rules, pd.DataFrame) else 0,
        "subcat_rules": int(len(model.subcat_rules)) if isinstance(model.subcat_rules, pd.DataFrame) else 0,
        "occasion_profiles": int(len(model.occasion_popularity)),
        "product_groups": int(len(set(model.product_group_map.values()))),
        "content_similarity": "loaded" if model.content_sim is not None else "not_loaded",
        "collaborative_similarity": "loaded" if model.collab_sim is not None else "not_loaded"
    }

    output_path = os.path.join(data_dir, "hybrid_recommender_model.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    summary_path = os.path.join(data_dir, "training_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(model.training_report, f, indent=2, ensure_ascii=False)

    print("Model saved successfully.")
    print(f"Model path: {output_path}")
    print(f"Summary path: {summary_path}")

    return model


if __name__ == "__main__":
    train_and_save_model(r"C:\D drive\sales_recommendation_system\data")