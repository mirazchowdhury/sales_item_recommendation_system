import os
import pickle
import traceback
import __main__

from flask import Flask, request, jsonify
from flasgger import Swagger

from model_trainer import HybridRecommenderModel


__main__.HybridRecommenderModel = HybridRecommenderModel


# =========================================================
# Flask App Setup
# =========================================================

app = Flask(__name__)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Hybrid Retail Recommendation Engine API",
        "description": "Retail recommendation API using basket rules, collaborative similarity, content similarity, and occasion fallback.",
        "version": "1.0.0"
    }
}

Swagger(app, config=swagger_config, template=swagger_template)


# =========================================================
# Config
# =========================================================

DATA_DIR = r"C:\D drive\sales_recommendation_system\data"
MODEL_PATH = os.path.join(DATA_DIR, "hybrid_recommender_model.pkl")

model = None


# =========================================================
# Model Loading
# =========================================================

def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully.")
    print(f"Model path: {MODEL_PATH}")

    if hasattr(model, "training_report"):
        print("Training report:")
        print(model.training_report)


# =========================================================
# Helper Functions
# =========================================================

def clean_item_id(value):
    if value is None:
        return None

    return str(value).strip()


def clean_quantity(value):
    try:
        quantity = int(value)

        if quantity <= 0:
            return 1

        return quantity

    except Exception:
        return 1


def build_input_items(items):
    cart_item_ids = []
    item_quantities = {}

    for item in items:
        item_id = clean_item_id(item.get("itemid"))
        quantity = clean_quantity(item.get("quantity", 1))

        if not item_id:
            continue

        cart_item_ids.append(item_id)
        item_quantities[item_id] = quantity

    return cart_item_ids, item_quantities


def build_recommendation_response(recommended_ids, model_insights):
    recommendations = []

    raw_scores = model_insights.get("raw_scores", {})

    for item_id in recommended_ids:
        meta = model.item_meta.get(item_id, {})

        item_name = meta.get("item_name", item_id)
        score = raw_scores.get(item_name, 0)

        recommendations.append({
            "category": meta.get("Category", ""),
            "item_name": item_name,
            "itemid": item_id,
            "score": score
        })

    return recommendations


# =========================================================
# Routes
# =========================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Hybrid Retail Recommendation Engine API is running.",
        "swagger": "http://127.0.0.1:5000/apidocs"
    })


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model is not loaded."
        }), 500

    return jsonify({
        "status": "ok",
        "message": "Model is loaded and API is healthy."
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    Recommend products based on customer cart
    ---
    tags:
      - Recommendation
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - customerid
            - date and time
            - items
          properties:
            customerid:
              type: string

            date and time:
              type: string

            items:
              type: array
              items:
                type: object
                required:
                  - itemid
                  - quantity
                properties:
                  itemid:
                    type: string

                  quantity:
                    type: integer

    responses:
      200:
        description: Product recommendations
        schema:
          type: object
          properties:
            input_item_names:
              type: array
              items:
                type: string
            recommendations:
              type: array
              items:
                type: object
                properties:
                  category:
                    type: string
                  item_name:
                    type: string
                  itemid:
                    type: string
                  score:
                    type: number
      400:
        description: Bad request
      500:
        description: Server error
    """

    try:
        if model is None:
            return jsonify({
                "error": "Model is not loaded."
            }), 500

        payload = request.get_json()

        if not payload:
            return jsonify({
                "error": "Invalid JSON payload."
            }), 400

        customer_id = payload.get("customerid")
        date_time = payload.get("date and time")
        items = payload.get("items", [])

        if not customer_id:
            return jsonify({
                "error": "customerid is required."
            }), 400

        if not date_time:
            return jsonify({
                "error": "date and time is required."
            }), 400

        if not isinstance(items, list) or len(items) == 0:
            return jsonify({
                "error": "items must be a non empty list."
            }), 400

        cart_item_ids, item_quantities = build_input_items(items)

        if not cart_item_ids:
            return jsonify({
                "error": "No valid itemid found in items."
            }), 400

        recommended_ids, model_insights = model.recommend(
            current_cart=cart_item_ids,
            top_n=5,
            customer_id=customer_id
        )

        recommendations = build_recommendation_response(
            recommended_ids=recommended_ids,
            model_insights=model_insights
        )

        response = {
            "input_item_names": model_insights.get("input_item_names", []),
            "recommendations": recommendations
        }

        return jsonify(response), 200

    except Exception as exc:
        print("Recommendation API error:")
        print(traceback.format_exc())

        return jsonify({
            "error": str(exc)
        }), 500


@app.route("/api/recommend/debug", methods=["POST"])
def recommend_debug():
    """
    Recommend products with model insight details
    ---
    tags:
      - Recommendation
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - customerid
            - date and time
            - items
          properties:
            customerid:
              type: string

            date and time:
              type: string

            items:
              type: array
              items:
                type: object
                properties:
                  itemid:
                    type: string

                  quantity:
                    type: integer

    responses:
      200:
        description: Product recommendations with model insights
    """

    try:
        if model is None:
            return jsonify({
                "error": "Model is not loaded."
            }), 500

        payload = request.get_json()

        if not payload:
            return jsonify({
                "error": "Invalid JSON payload."
            }), 400

        customer_id = payload.get("customerid")
        date_time = payload.get("date and time")
        items = payload.get("items", [])

        if not customer_id:
            return jsonify({
                "error": "customerid is required."
            }), 400

        if not date_time:
            return jsonify({
                "error": "date and time is required."
            }), 400

        if not isinstance(items, list) or len(items) == 0:
            return jsonify({
                "error": "items must be a non empty list."
            }), 400

        cart_item_ids, item_quantities = build_input_items(items)

        recommended_ids, model_insights = model.recommend(
            current_cart=cart_item_ids,
            top_n=5,
            customer_id=customer_id
        )

        recommendations = build_recommendation_response(
            recommended_ids=recommended_ids,
            model_insights=model_insights
        )

        response = {
            "input_item_names": model_insights.get("input_item_names", []),
            "recommendations": recommendations,
            "model_insights": model_insights
        }

        return jsonify(response), 200

    except Exception as exc:
        print("Debug recommendation API error:")
        print(traceback.format_exc())

        return jsonify({
            "error": str(exc)
        }), 500


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    load_model()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )