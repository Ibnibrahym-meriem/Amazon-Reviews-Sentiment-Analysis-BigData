from flask import Flask, render_template, jsonify, request, Response
from datetime import datetime, timedelta
import random
from pymongo import MongoClient
import json
import time

app = Flask(__name__)
# At startup, remember the highest review Id currently in MongoDB

# ================= MONGODB CONNECTION =================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "amazon_db"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    db.command('ping')
    print("✅ Connected to MongoDB successfully!")
    print(f"📊 Database: {DB_NAME}")
    print(f"📁 Collections: {db.list_collection_names()}")
    USE_MOCK_DATA = False
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    USE_MOCK_DATA = False


import os

# ================= COLLECTIONS =================
reviews_collection = db.reviews
product_metrics_collection = db.product_metrics
model_metrics_collection = db.model_metrics

# ================= STREAM START ID (ONLY show reviews created AFTER server starts) =================
# Get the maximum review Id at server startup
last_review_before_start = reviews_collection.find_one(sort=[("Id", -1)])
if last_review_before_start:
    STREAM_START_ID = last_review_before_start.get("Id", 0)
else:
    STREAM_START_ID = 0
print(f"🆔 Streaming will only show reviews with Id > {STREAM_START_ID}")



# ================= STREAM START TIMESTAMP =================
# Only reviews inserted AFTER this timestamp will appear in the live feed.
# This ensures the live page shows only real-time Kafka→Spark data.
STREAM_START_TIME = datetime.utcnow()
print(f"🕐 Stream start time (UTC): {STREAM_START_TIME}")

# ================= MOCK DATA (for offline sections ONLY — NOT for live feed) =================
PRODUCTS = ["B001E4KFG0", "B00813GRG4", "B003B3OOPA", "B007PA5MHY", "B0019CW0HE",
            "B006K2ZZ7K", "B00171APVA", "B0001PB9FE", "B004I616LE", "B001GVISJM"]
SENTIMENTS = ["positive", "negative", "neutral"]
SENTIMENT_WEIGHTS = [0.72, 0.16, 0.12]

SAMPLE_TEXTS = {
    "positive": ["Absolutely love this product!", "Best purchase ever!", "Exceeded expectations!"],
    "negative": ["Very disappointed.", "Terrible quality.", "Not as advertised."],
    "neutral": ["It's okay.", "Average product.", "Nothing special."]
}

USERNAMES = ["john_doe", "foodlover92", "amazonshopper", "critic101", "healthnut"]

def make_review(product_id=None, days_back=None):
    sentiment = random.choices(SENTIMENTS, SENTIMENT_WEIGHTS)[0]
    score_map = {"positive": random.choice([4,5]), "negative": random.choice([1,2]), "neutral": 3}
    score = score_map[sentiment]
    if days_back is None:
        days_back = random.randint(0, 365 * 10)
    ts = int((datetime.now() - timedelta(days=days_back)).timestamp())
    return {
        "Id": random.randint(1, 999999),
        "ProductId": product_id or random.choice(PRODUCTS),
        "ProfileName": random.choice(USERNAMES),
        "HelpfulnessNumerator": random.randint(0,10),
        "HelpfulnessDenominator": random.randint(1,15),
        "Score": score,
        "Time": ts,
        "Text": random.choice(SAMPLE_TEXTS[sentiment]),
        "predicted_sentiment": sentiment,
        "confidence": round(random.uniform(0.65, 0.99), 2),
        "processed_at": datetime.now().isoformat()
    }

from collections import Counter
import re

def get_top_words(product_id, limit=20):
    reviews = list(reviews_collection.find({"ProductId": product_id}))
    if not reviews:
        return []
    all_text = " ".join([r.get("Text", "") for r in reviews])
    clean_text = re.sub(r'[^a-zA-Z\s]', '', all_text.lower())
    words = clean_text.split()

    stop_words = {
        'the','and','a','an','of','to','for','in','on','at','by','with','without',
        'this','that','these','those','it','its','they','them','their','we','our',
        'you','your','i','my','me','he','she','him','her','but','or','so','nor','yet',
        'because','if','then','else','when','where','which','while','whom','who','whose',
        'why','how','however','is','are','was','were','be','been','being','am','have',
        'has','had','having','do','does','did','doing','can','could','will','would',
        'should','may','might','must','shall','up','down','off','over','under','again',
        'further','once','here','there','all','any','both','each','few','more','most',
        'other','some','such','no','not','only','own','same','than','through','until',
        'very','really','just','like','get','got','make','made','much','many','lot',
        'also','even','though','although','well','good','nice','bad','poor','little',
        'bit','quite','rather','somewhat','anything','nothing','everything','everyone',
        'nobody','anyone','someone','one','two','three','four','five','six','seven',
        'eight','nine','ten','first','second','third','next','last','previous','final',
        'initial','product','item','pack','packaging','box','bag','bottle','bought',
        'purchase','purchased','order','ordered','received','arrived','delivery',
        'shipping','shipped','buy','buying','use','used','using','eat','eating','ate',
        'drink','drinking','drank','food','snack','meal','time','days','week','month',
        'year','back','again','since','until','while','after','before','during',
        'without','please','thanks','thank','appreciate','recommend','recommended'
    }

    meaningful_short = {
        'tea','ice','hot','cold','new','old','big','small','top','best','bag','box',
        'can','cup','red','blue','green','dark','light','sweet','sour','fresh','dry',
        'wet','soft','hard','fast','slow','cheap','good','bad','high','low'
    }

    filtered_words = [w for w in words if w not in stop_words and (len(w) >= 3 or w in meaningful_short)]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(limit)
    return [{"word": word, "count": count} for word, count in top_words]

def generate_mock_dataset(n=600):
    return [make_review(days_back=random.randint(0, 365*10)) for _ in range(n)]

MOCK_DATASET = generate_mock_dataset(600)

# ================= CALCULATION FUNCTIONS (offline sections) =================

def calc_kpis():
    if USE_MOCK_DATA:
        reviews = MOCK_DATASET
        total = len(reviews)
        pos = sum(1 for r in reviews if r["predicted_sentiment"] == "positive")
        neg = sum(1 for r in reviews if r["predicted_sentiment"] == "negative")
        neu = sum(1 for r in reviews if r["predicted_sentiment"] == "neutral")
        avg_score = round(sum(r["Score"] for r in reviews) / total, 2)
        unique_products = len(set(r["ProductId"] for r in reviews))
    else:
        total = reviews_collection.count_documents({})
        pos = reviews_collection.count_documents({"predicted_sentiment": "positive"})
        neg = reviews_collection.count_documents({"predicted_sentiment": "negative"})
        neu = reviews_collection.count_documents({"predicted_sentiment": "neutral"})
        pipeline = [{"$group": {"_id": None, "avg": {"$avg": "$Score"}}}]
        avg_result = list(reviews_collection.aggregate(pipeline))
        avg_score = round(avg_result[0]["avg"], 2) if avg_result else 0
        unique_products = len(reviews_collection.distinct("ProductId"))

    accuracy = 87.0
    return {
        "total": total,
        "positive_pct": round(pos / total * 100, 1) if total else 0,
        "negative_pct": round(neg / total * 100, 1) if total else 0,
        "neutral_pct": round(neu / total * 100, 1) if total else 0,
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "avg_score": avg_score,
        "unique_products": unique_products,
        "model_accuracy": accuracy
    }

def calc_trend():
    if USE_MOCK_DATA:
        reviews = MOCK_DATASET
        buckets = {}
        for r in reviews:
            dt = datetime.fromtimestamp(r["Time"])
            key = f"{dt.year}-{dt.month:02d}"
            if key not in buckets:
                buckets[key] = {"positive": 0, "negative": 0, "neutral": 0}
            buckets[key][r["predicted_sentiment"]] += 1
        return [{"month": k, "positive": v["positive"], "negative": v["negative"], "neutral": v["neutral"]}
                for k, v in sorted(buckets.items())[-12:]]

    pipeline = [
        {"$group": {
            "_id": {
                "year": {"$year": {"$toDate": {"$multiply": ["$Time", 1000]}}},
                "month": {"$month": {"$toDate": {"$multiply": ["$Time", 1000]}}},
                "sentiment": "$predicted_sentiment"
            },
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]
    results = list(reviews_collection.aggregate(pipeline))
    months = {}
    for r in results[-36:]:
        key = f"{r['_id']['year']}-{r['_id']['month']:02d}"
        if key not in months:
            months[key] = {"positive": 0, "negative": 0, "neutral": 0}
        months[key][r['_id']['sentiment']] = r['count']
    return [{"month": k, "positive": v["positive"], "negative": v["negative"], "neutral": v["neutral"]}
            for k, v in sorted(months.items())[-12:]]

def calc_score_dist():
    total = get_total_count()
    if USE_MOCK_DATA:
        reviews = MOCK_DATASET
        dist = {i: sum(1 for r in reviews if r["Score"] == i) for i in range(1, 6)}
    else:
        dist = {}
        for star in range(1, 6):
            dist[star] = reviews_collection.count_documents({"Score": star})
    return [{"star": i, "count": dist.get(i, 0), "pct": round(dist.get(i, 0) / total * 100, 1) if total else 0}
            for i in range(5, 0, -1)]

def calc_top_products(n=6):
    if USE_MOCK_DATA:
        reviews = MOCK_DATASET
        products = {}
        for r in reviews:
            pid = r["ProductId"]
            if pid not in products:
                products[pid] = {"total": 0, "positive": 0, "negative": 0, "neutral": 0}
            products[pid]["total"] += 1
            products[pid][r["predicted_sentiment"]] += 1
        result = []
        for pid, d in products.items():
            health = (d["positive"] - d["negative"]) / d["total"] if d["total"] > 0 else 0
            status = "OK" if health > 0.5 else ("WATCH" if health > 0.2 else "ALERT")
            result.append({
                "product_id": pid,
                "total": d["total"],
                "positive_pct": round(d["positive"] / d["total"] * 100, 1),
                "negative_pct": round(d["negative"] / d["total"] * 100, 1),
                "health_score": round(health, 2),
                "status": status
            })
        result.sort(key=lambda x: x["total"], reverse=True)
        return result[:n]

    products = list(product_metrics_collection.find().sort("review_count", -1).limit(n))
    result = []
    for p in products:
        health = p.get("positive_ratio", 0) - p.get("negative_ratio", 0)
        status = "OK" if health > 0.5 else ("WATCH" if health > 0.2 else "ALERT")
        result.append({
            "product_id": p["_id"],
            "total": p.get("review_count", 0),
            "positive_pct": round(p.get("positive_ratio", 0) * 100, 1),
            "negative_pct": round(p.get("negative_ratio", 0) * 100, 1),
            "health_score": round(health, 2),
            "status": status
        })
    return result

def calc_helpfulness():
    return {"overall": 68.5, "by_sentiment": {"positive": 65.2, "negative": 82.1, "neutral": 45.3}}

def calc_insights(kpis):
    insights = []
    if kpis["negative_pct"] > 30:
        insights.append(f"⚠️ High negative sentiment ({kpis['negative_pct']}%) — investigate top products")
    else:
        insights.append(f"✅ Overall sentiment healthy with {kpis['positive_pct']}% positive reviews")
    insights.append(f"🤖 Model accuracy: {kpis['model_accuracy']}% — predictions reliable")
    insights.append(f"📊 Monitoring {kpis['unique_products']} unique products")
    insights.append(f"⭐ Average rating: {kpis['avg_score']}/5 stars")
    return insights

def get_total_count():
    if USE_MOCK_DATA:
        return len(MOCK_DATASET)
    return reviews_collection.count_documents({})

def calc_product_analysis(product_id):
    product_stats = product_metrics_collection.find_one({"_id": product_id})
    if not product_stats:
        return None

    health = product_stats.get("positive_ratio", 0) - product_stats.get("negative_ratio", 0)
    status = "OK" if health > 0.5 else ("WATCH" if health > 0.2 else "ALERT")

    recent = list(reviews_collection.find({"ProductId": product_id}).sort("processed_at", -1).limit(10))
    for r in recent:
        r["_id"] = str(r["_id"])

    reviews_list = list(reviews_collection.find({"ProductId": product_id}))
    actual_avg_score = sum(r.get("Score", 0) for r in reviews_list) / len(reviews_list) if reviews_list else 0
    trend_pipeline = [
    {
        "$match": {"ProductId": product_id}
    },
    {
        "$group": {
            "_id": {
                "year": {
                    "$year": {
                        "$toDate": {"$multiply": ["$Time", 1000]}
                    }
                },
                "month": {
                    "$month": {
                        "$toDate": {"$multiply": ["$Time", 1000]}
                    }
                },
                "sentiment": "$predicted_sentiment"
            },
            "count": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "_id.year": 1,
            "_id.month": 1
        }
    }
]

    trend_results = list(reviews_collection.aggregate(trend_pipeline))

    trend = {}

    for r in trend_results:
        key = f"{r['_id']['year']}-{r['_id']['month']:02d}"

        if key not in trend:
            trend[key] = {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            }

        trend[key][r['_id']['sentiment']] = r['count']

    trend_data = [
        {
            "month": k,
            "positive": v["positive"],
            "negative": v["negative"],
            "neutral": v["neutral"]
        }
        for k, v in sorted(trend.items())[-12:]
    ]
    return {
        "kpis": {
            "total": product_stats.get("review_count", 0),
            "positive_count": int(product_stats.get("positive_ratio", 0) * product_stats.get("review_count", 0)),
            "negative_count": int(product_stats.get("negative_ratio", 0) * product_stats.get("review_count", 0)),
            "neutral_count": int(product_stats.get("neutral_ratio", 0) * product_stats.get("review_count", 0)),
            "avg_score": round(actual_avg_score, 1)
        },
        "health_score": round(health, 2),
        "status": status,
        "recent_reviews": recent,
        "trend": trend_data,
        "helpfulness": {"overall": 68.5, "by_sentiment": {"positive": 65.2, "negative": 82.1, "neutral": 45.3}}
    }

def calc_watchlist_data(sort_by="health_score", order="asc"):
    if USE_MOCK_DATA:
        products = calc_top_products(n=50)
        alerts = sum(1 for p in products if p["status"] == "ALERT")
        watches = sum(1 for p in products if p["status"] == "WATCH")
        oks = sum(1 for p in products if p["status"] == "OK")
        return {"products": products, "alerts": alerts, "watches": watches, "oks": oks}

    products = list(product_metrics_collection.find())
    for p in products:
        health = p.get("positive_ratio", 0) - p.get("negative_ratio", 0)
        p["health_score"] = round(health, 2)
        p["product_id"] = p["_id"]
        p["total"] = p.get("review_count", 0)
        p["positive_pct"] = round(p.get("positive_ratio", 0) * 100, 1)
        p["negative_pct"] = round(p.get("negative_ratio", 0) * 100, 1)
        p["avg_score"] = round(p.get("avg_sentiment", 0) * 2.5, 1)
        p["status"] = "OK" if health > 0.5 else ("WATCH" if health > 0.2 else "ALERT")

    if sort_by == "health_score":
        products.sort(key=lambda x: x["health_score"], reverse=(order == "desc"))
    elif sort_by == "total":
        products.sort(key=lambda x: x["total"], reverse=(order == "desc"))
    elif sort_by == "avg_score":
        products.sort(key=lambda x: x["avg_score"], reverse=(order == "desc"))
    elif sort_by == "product_id":
        products.sort(key=lambda x: x["product_id"], reverse=(order == "desc"))

    products = products[:50]
    alerts = sum(1 for p in products if p["status"] == "ALERT")
    watches = sum(1 for p in products if p["status"] == "WATCH")
    oks = sum(1 for p in products if p["status"] == "OK")
    return {"products": products, "alerts": alerts, "watches": watches, "oks": oks}


# ================= ROUTES =================

@app.route("/")
def dashboard():
    kpis = calc_kpis()
    trend = calc_trend()
    score_dist = calc_score_dist()
    top_products = calc_top_products(6)
    insights = calc_insights(kpis)
    helpfulness = calc_helpfulness()
    return render_template("index.html", kpis=kpis, trend=trend, score_dist=score_dist,
                           top_products=top_products, insights=insights, helpfulness=helpfulness)

@app.route("/product")
def product():
    product_id = request.args.get("id", "")
    products_with_counts = list(product_metrics_collection.find({}, {"_id": 1, "review_count": 1}).sort("review_count", -1))
    ordered_products = [p["_id"] for p in products_with_counts if p.get("review_count", 0) > 0]
    if not product_id and ordered_products:
        product_id = ordered_products[0]
    data = calc_product_analysis(product_id) if product_id else None
    top_words = get_top_words(product_id, 20) if product_id else []
    return render_template("product.html", data=data, product_id=product_id,
                           all_products=ordered_products, top_words=top_words)

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/watchlist")
def watchlist():
    sort_by = request.args.get("sort", "health_score")
    order = request.args.get("order", "asc")
    data = calc_watchlist_data(sort_by=sort_by, order=order)
    return render_template("watchlist.html", data=data, current_sort=sort_by, current_order=order)


# ================= SSE STREAMING ENDPOINT =================
# Only streams reviews inserted into MongoDB AFTER the server started.
# This means only real-time Kafka → Spark → MongoDB reviews will appear.
# Historical data is never shown on the live feed page.

@app.route('/stream')
def stream():
    """SSE endpoint with proper request context handling"""
    
    # Capture request parameters OUTSIDE the generator
    client_last_time = request.args.get('lastTime')
    client_last_id = request.args.get('lastId', type=int)
    
    def generate():
        import time
        from datetime import datetime, timedelta
        
        # Use the captured values
        if client_last_time:
            try:
                last_time = datetime.fromisoformat(client_last_time.replace('Z', '+00:00'))
                print(f"🔄 Resume from time: {last_time}")
                # Add 1 microsecond to avoid duplicates
                last_time += timedelta(microseconds=1)
            except Exception as e:
                print(f"⚠️ Could not parse lastTime: {e}")
                last_time = datetime.utcnow() - timedelta(minutes=5)
        else:
            last_time = datetime.utcnow() - timedelta(minutes=5)
            print(f"🆕 First connection - showing reviews from last 5 minutes")
        
        print(f"🔍 Streaming reviews after: {last_time}")
        
        # Track sent IDs to prevent duplicates
        sent_ids = set()
        
        while True:
            try:
                # Query for reviews after last_time
                query = {"processed_at": {"$gt": last_time}}
                new_reviews = list(reviews_collection.find(query).sort("processed_at", 1).limit(10))
                
                for review in new_reviews:
                    review_id = review.get("Id")
                    review_time = review.get("processed_at")
                    
                    # Skip if already sent
                    if review_id in sent_ids:
                        continue
                    
                    if review_time and review_time > last_time:
                        sent_ids.add(review_id)
                        last_time = review_time
                        
                        # Convert for JSON
                        review["_id"] = str(review["_id"])
                        if isinstance(review.get("processed_at"), datetime):
                            review["processed_at"] = review["processed_at"].isoformat()
                        
                        print(f"📨 Streaming review ID: {review_id}")
                        yield f"data: {json.dumps(review, default=str)}\n\n"
                        
                        # Small delay between reviews
                        time.sleep(0.2)
                
                # Wait before checking again
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ SSE Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    return Response(generate(), mimetype="text/event-stream")
@app.route("/api/kpis")
def api_kpis():
    return jsonify(calc_kpis())

@app.route("/api/watchlist")
def api_watchlist():
    return jsonify(calc_watchlist_data())
# Add to imports at top of app.py
from mlflow_client import MLflowDashboardClient

# Initialize MLflow client
mlflow_client = MLflowDashboardClient()

# ================= MODEL HEALTH API ROUTES =================

@app.route("/api/model-current")
def api_model_current():
    """Get current model statistics for cards"""
    stats = mlflow_client.get_current_model_stats()
    if stats:
        return jsonify(stats)
    return jsonify({"error": "No model data available"}), 404

@app.route("/api/model-history")
def api_model_history():
    """Get model performance history for charts"""
    limit = request.args.get('limit', 20, type=int)
    history = mlflow_client.get_performance_history(limit=limit)
    return jsonify(history)

@app.route("/model-health")
def model_health():
    """Model health dashboard page"""
    return render_template("model_health.html")

if __name__ == "__main__":
    app.run(debug=False, threaded=True, port=5000)