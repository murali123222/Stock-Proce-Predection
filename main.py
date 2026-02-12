# ================== IMPORTS ==================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

import smtplib
from email.message import EmailMessage

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn


# ================== APP SETUP ==================
app = FastAPI(title="Stock LSTM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== EMAIL CONFIG ==================
# 🔴 EDIT THESE
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "your_email@gmail.com"           # your gmail
SMTP_PASSWORD = "your_app_password_here"     # gmail app password
CONTACT_TO_EMAIL = "your_email@gmail.com"    # where mail should arrive


# ================== CONTACT MODEL ==================
class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    message: str


# ================== LSTM MODEL ==================
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ================== UTIL FUNCTIONS ==================
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


# ================== PREDICTION ENDPOINT ==================
@app.get("/predict_lstm/{ticker}")
def predict_lstm_price(
    ticker: str,
    period: str = "1y",
    seq_length: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
):
    # 1️⃣ Download stock data
    try:
        df = yf.download(ticker.upper(), period=period, interval="1d", progress=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Download failed: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail="Invalid ticker or no data found")

    close_prices = df["Close"].values.reshape(-1, 1).astype(np.float32)

    if len(close_prices) <= seq_length:
        raise HTTPException(status_code=400, detail="Not enough historical data")

    # 2️⃣ Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # 3️⃣ Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StockLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4️⃣ Train
    model.train()
    for _ in range(max(1, epochs)):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 5️⃣ Predict next day
    model.eval()
    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    last_tensor = torch.tensor(last_seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        scaled_pred = model(last_tensor).cpu().numpy()

    predicted_price = float(scaler.inverse_transform(scaled_pred)[0][0])
    last_close = float(close_prices[-1][0])

    return {
        "ticker": ticker.upper(),
        "last_close_price": last_close,
        "predicted_next_close_price": predicted_price,
        "epochs_used": epochs,
    }


# ================== CONTACT ENDPOINT ==================
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "thotamurali444@gmail.com"
SMTP_PASSWORD = "sezz rcis sjuk ghqn"
CONTACT_TO_EMAIL = "thotamurali444@gmail.com"

@app.post("/contact")
def submit_contact(form: ContactRequest):

    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        raise HTTPException(status_code=500, detail="Email not configured")

    msg = EmailMessage()
    msg["Subject"] = f"Contact from {form.name}"
    msg["From"] = SMTP_USER
    msg["To"] = CONTACT_TO_EMAIL

    msg.set_content(
        f"Name: {form.name}\n"
        f"Email: {form.email}\n\n"
        f"Message:\n{form.message}"
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Email send failed: {e}")

    return {"status": "ok", "message": "Message sent successfully"}
