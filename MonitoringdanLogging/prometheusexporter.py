from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import requests
import random

# 3 Metrics minimal
REQUEST_TOTAL = Counter("model_requests_total", "Total requests")
REQUEST_DURATION = Histogram("model_request_duration_seconds", "Request latency")
CHURN_PREDICTION_RATIO = Gauge("churn_prediction_ratio", "Ratio of churn predictions")


def predict_churn():
    """Simulate prediction"""
    start = time.time()

    # Simulate API call
    try:
        resp = requests.post(
            "http://127.0.0.1:5001/invocations",
            json=[[random.random() for _ in range(30)]],
            timeout=5,
        )
        latency = time.time() - start
        churn_prob = random.random() > 0.7  # simulate 30% churn

        REQUEST_TOTAL.inc()
        REQUEST_DURATION.observe(latency)
        CHURN_PREDICTION_RATIO.set(0.3 if churn_prob else 0.0)

        print(
            f"✅ Prediction: {'Churn' if churn_prob else 'No Churn'} (latency: {latency:.3f}s)"
        )
    except:
        print("❌ API error")


if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus metrics on :8000/metrics")

    while True:
        predict_churn()
        time.sleep(10)
