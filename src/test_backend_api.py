import urllib.request
import json

def test_endpoint(name, url, method="GET", payload=None):
    try:
        data = json.dumps(payload).encode("utf-8") if payload else None
        headers = {"Content-Type": "application/json"} if payload else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
            print(f"[PASS] {name:<36} | Status {resp.getcode()} | Success: {body.get('success')}")
            return body
    except Exception as e:
        print(f"[FAIL] {name:<36} | Error: {e}")
        return None

base = "http://localhost:3001/api"
print("=" * 65)
print("TESTING BACKEND API ENDPOINTS (http://localhost:3001/api)")
print("=" * 65)

test_endpoint("GET /api/health", f"{base}/health")
test_endpoint("GET /api/dataset/summary", f"{base}/dataset/summary")
test_endpoint("GET /api/dataset/preview?limit=3", f"{base}/dataset/preview?limit=3")
test_endpoint("GET /api/models/performance", f"{base}/models/performance")
test_endpoint("GET /api/models/confusion-matrix", f"{base}/models/Random%20Forest/confusion-matrix")
test_endpoint("GET /api/features/importance", f"{base}/features/importance")

print("\n" + "=" * 65)
print("TESTING REAL ML PREDICTION VIA BACKEND -> ML SERVICE")
print("=" * 65)

# Prime Applicant
prime_payload = {
    "model_name": "Random Forest",
    "inputs": {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 6500.0,
        "CoapplicantIncome": 2500.0,
        "LoanAmount": 140.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Semiurban"
    }
}
prime_res = test_endpoint("POST /api/predict (Prime Applicant)", f"{base}/predict", method="POST", payload=prime_payload)
if prime_res:
    data = prime_res["data"]
    print(f"       -> Decision: {data['prediction']} | Approval Prob: {data['approval_probability']:.2%} | Risk: {data['risk_level']}")

# High-Risk Applicant
risk_payload = {
    "model_name": "Random Forest",
    "inputs": {
        "Gender": "Male",
        "Married": "No",
        "Dependents": "0",
        "Education": "Not Graduate",
        "Self_Employed": "Yes",
        "ApplicantIncome": 2000.0,
        "CoapplicantIncome": 0.0,
        "LoanAmount": 250.0,
        "Loan_Amount_Term": 180.0,
        "Credit_History": 0.0,
        "Property_Area": "Rural"
    }
}
risk_res = test_endpoint("POST /api/predict (High-Risk Applicant)", f"{base}/predict", method="POST", payload=risk_payload)
if risk_res:
    data = risk_res["data"]
    print(f"       -> Decision: {data['prediction']} | Approval Prob: {data['approval_probability']:.2%} | Risk: {data['risk_level']}")

print("\n" + "=" * 65)
print("TESTING SUPABASE PREDICTIONS ENDPOINT")
print("=" * 65)
sup_res = test_endpoint("GET /api/predictions", f"{base}/predictions")
if sup_res:
    print(f"       -> Configured: {sup_res['data'].get('configured')} | Rows: {len(sup_res['data'].get('rows', []))}")
