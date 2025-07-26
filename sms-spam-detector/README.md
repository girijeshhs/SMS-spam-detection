# Email Spam Detector Project

## Structure

- `backend/` — Flask API and ML model
- `frontend/` — Simple HTML/JS frontend
- `spam.csv` — Your dataset

## How to Run

### 1. Backend
```bash
cd sms-spam-detector/backend
pip install -r requirements.txt
python app.py
```

### 2. Frontend
Just open `frontend/index.html` in your browser.

## Usage
- Paste email text in the box and click "Check Spam".
- The backend will predict if the email is spam or not.

---

You can retrain the model by sending a POST request to `/train` endpoint.
