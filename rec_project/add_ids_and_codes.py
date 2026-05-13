import pandas as pd
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = 'recommendations_output_gpu.csv'
OUTPUT_FILE = 'final_recommendations_with_ids.csv'


def add_recommendation_details():
    if not pd.io.common.file_exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} file-ti paowa jayni!")
        return

    print("Step 1: Reading Dataset...")
    df = pd.read_csv(INPUT_FILE)

    # Nam theke ID ebong Code khujar jonno ekta mapping dictionary toiri kora
    # Jate search process fast hoy
    name_to_id = dict(zip(df['item_name'], df['item_id']))
    name_to_code = dict(zip(df['item_name'], df['product_code']))

    rec_ids_list = []
    rec_codes_list = []

    print("Step 2: Mapping Names to IDs and Product Codes...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        # Existing top_5_recommendations column theke nam gulo split kora
        # Ekhane ' | ' delimiter bebohar kora hoyeche jeta apnar file-e ache
        rec_names = str(row['top_5_recommendations']).split(' | ')

        ids = []
        codes = []

        for name in rec_names:
            name = name.strip()
            # Mapping dictionary theke ID ebong Code khuje ber kora
            item_id = name_to_id.get(name, "N/A")
            prod_code = name_to_code.get(name, "N/A")

            ids.append(str(item_id))
            codes.append(str(prod_code))

        # Pipe separator diye join kora
        rec_ids_list.append(" | ".join(ids))
        rec_codes_list.append(" | ".join(codes))

    # Notun column add kora
    df['recommendations_item_ids'] = rec_ids_list
    df['recommendations_product_codes'] = rec_codes_list

    # Final CSV Save
    print(f"Step 3: Saving Final File...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- Success! Final file saved as: {OUTPUT_FILE} ---")


if __name__ == "__main__":
    add_recommendation_details()