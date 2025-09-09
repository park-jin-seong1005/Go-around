import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
import joblib

# === 모델 정의 ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=6):
        super(Autoencoder, self).__init__()
        negative_slope = 0.01
        dropout_rate = 0.05

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),

            nn.Linear(16, encoding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),

            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout_rate),

            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 안정적인 시그모이드

def detect_go_around_probability(model_path, test_data_path, scaler_path=None, threshold_mse=0.1):
    # CSV 로드
    df = pd.read_csv(test_data_path)
    df = df.select_dtypes(include=[np.number]).fillna(df.mean())

    # 스케일러
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = RobustScaler()
        scaler.fit(df)

    data_scaled = scaler.transform(df)
    input_dim = data_scaled.shape[1]

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model = Autoencoder(input_dim, checkpoint.get('encoding_dim', 6)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = Autoencoder(input_dim, 6).to(device)
        model.load_state_dict(checkpoint)

    model.eval()
    tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(tensor)
        mse = torch.mean((tensor - output) ** 2, dim=1).cpu().numpy()

    probabilities = [sigmoid((m - threshold_mse) * 5) for m in mse]

    return probabilities  # 모든 행의 확률 리스트 반환


# === 실행 ===
if __name__ == "__main__":
    model_path = "./best_autoencoder.pt"
    test_data_path = "./output/5904개중 59개인데 고어라운드 안된거(valid)(로그변환).csv"
    scaler_path = "./scaler.pkl"
    threshold_mse = 0.1
    
    output_path = r"./output/output1.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        probs = detect_go_around_probability(model_path, test_data_path, scaler_path, threshold_mse)
        
        # 각 행 출력
        for i, p in enumerate(probs, start=1):
            print(f"{i}행 고어라운드 확률: {p:.6f}")

        # CSV 저장
        pd.DataFrame({"row": range(1, len(probs)+1), "probability": probs}).to_csv(output_path, index=False)
        print(f"✅ 분석 완료. 결과 저장: {output_path}")

    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
