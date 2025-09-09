import os
import pandas as pd
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import gc


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df = df.apply(pd.to_numeric, errors='coerce')

    print(f"📊 데이터 로드 완료 - Shape: {df.shape}")
    print(f"📊 로그 변환은 이미 적용된 상태입니다.")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️ 결측치 {missing_count}개 발견, 제거합니다.")
        df = df.dropna()
    else:
        print("✅ 결측치 없음")
    
    inf_count = np.isinf(df.values).sum()
    if inf_count > 0:
        print(f"⚠️ 무한값 {inf_count}개 발견, 제거합니다.")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    else:
        print("✅ 무한값 없음")

    print("🔄 RobustScaler로 정규화 진행중...")
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df)

    X_train, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42, shuffle=True)

    print(f"📊 학습 데이터: {X_train.shape}")
    print(f"📊 검증 데이터: {X_test.shape}")
    
    return X_train, X_test, scaler

# 기존 표준 오토인코더 (개선된 버전)
class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=4):
        super(ImprovedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Encoder: 12 ➔ 8 ➔ 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(8, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU()
        )

        # Decoder: 4 ➔ 8 ➔ 12
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


# 변분 오토인코더 (VAE)
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 12 ➔ 8
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 평균과 분산
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        # Decoder: 4 ➔ 8 ➔ 12
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(8, input_dim)
        )

    def encode(self, x):
        h = self.encoder_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar, z



# 베타-VAE (더 강한 정규화)
class BetaVAE(VariationalAutoencoder):
    def __init__(self, input_dim, latent_dim=4, beta=1.0):
        super(BetaVAE, self).__init__(input_dim, latent_dim)
        self.beta = beta


# 앙상블 오토인코더
class EnsembleAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=3, n_models=3):
        super(EnsembleAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.n_models = n_models
        
        # 다양한 구조의 오토인코더들
        self.autoencoders = nn.ModuleList()
        
        for i in range(n_models):
            # 각 모델마다 조금씩 다른 구조
            latent_dim = encoding_dim + i
            ae = ImprovedAutoencoder(input_dim, latent_dim)
            self.autoencoders.append(ae)
        
    def forward(self, x):
        outputs = []
        encodings = []
        
        for ae in self.autoencoders:
            decoded, encoded = ae(x)
            outputs.append(decoded)
            encodings.append(encoded)
            
        # 평균 출력
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output, encodings


# 손실 함수들
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE 손실 함수"""
    # 재구성 손실
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL 발산 손실
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss


def ensemble_loss(outputs, x, encodings=None):
    """앙상블 손실 함수"""
    losses = []
    for output in outputs:
        loss = F.mse_loss(output, x)
        losses.append(loss)
    
    # 평균 손실 + 다양성 손실
    avg_loss = torch.mean(torch.stack(losses))
    
    # 다양성 손실 (인코딩들이 너무 비슷하지 않도록)
    diversity_loss = 0.0
    if encodings and len(encodings) > 1:
        for i in range(len(encodings)):
            for j in range(i+1, len(encodings)):
                similarity = F.cosine_similarity(encodings[i], encodings[j], dim=1).mean()
                diversity_loss += similarity
        diversity_loss = diversity_loss / (len(encodings) * (len(encodings) - 1) / 2)
    
    return avg_loss + 0.1 * diversity_loss


def train_autoencoder(data_path, batch_size=64, epochs=500, lr=3e-5, model_type="improved"):
    """
    개선된 오토인코더 훈련 함수
    
    Args:
        model_type: "improved", "vae", "beta_vae", "ensemble" 중 선택
    """
    print(f"🎯 실험 시작: Model={model_type}, Batch={batch_size}, LR={lr}, Epochs={epochs}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📟 Using device: {device}")

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    try:
        X_train, X_test, scaler = load_and_preprocess_data(data_path)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size, drop_last=True)

        input_dim = X_train.shape[1]
        encoding_dim = 4  # 더 강한 압축

        print(f"📊 Input dimension: {input_dim}")
        print(f"📊 Encoding dimension: {encoding_dim}")
        print(f"📊 Model type: {model_type}")

        # 모델 선택
        if model_type == "improved":
            model = ImprovedAutoencoder(input_dim, encoding_dim).to(device)
            criterion = nn.MSELoss()
        elif model_type == "vae":
            model = VariationalAutoencoder(input_dim, encoding_dim).to(device)
            criterion = None  # VAE는 별도 손실 함수 사용
        elif model_type == "beta_vae":
            model = BetaVAE(input_dim, encoding_dim, beta=2.0).to(device)
            criterion = None  # Beta-VAE는 별도 손실 함수 사용
        elif model_type == "ensemble":
            model = EnsembleAutoencoder(input_dim, encoding_dim, n_models=3).to(device)
            criterion = None  # 앙상블은 별도 손실 함수 사용
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5, verbose=False
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", f"{model_type}_bs{batch_size}_lr{lr}_ep{epochs}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        model_save_path = os.path.join(log_dir, "best_autoencoder.pt")
        scaler_save_path = os.path.join(log_dir, "scaler.pkl")

        best_val_loss = float('inf')
        patience = 15
        wait = 0

        print(f"🚀 학습 시작 - Model: {model_type}, Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")
        print("="*70)

        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch[0]
                
                # 모델 타입에 따른 손실 계산
                if model_type == "improved":
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, inputs)
                elif model_type in ["vae", "beta_vae"]:
                    outputs, mu, logvar, _ = model(inputs)
                    beta = model.beta if hasattr(model, 'beta') else 1.0
                    loss = vae_loss(outputs, inputs, mu, logvar, beta)
                elif model_type == "ensemble":
                    outputs, encodings = model(inputs)
                    loss = F.mse_loss(outputs, inputs)  # 간단한 MSE 손실 사용
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # 메모리 정리
                del inputs, loss
                if model_type in ["vae", "beta_vae"]:
                    del outputs, mu, logvar
                elif model_type == "ensemble":
                    del outputs, encodings
                else:
                    del outputs
            
            train_loss = train_loss / train_batches if train_batches > 0 else 0.0

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    inputs = batch[0]
                    
                    # 모델 타입에 따른 손실 계산
                    if model_type == "improved":
                        outputs, _ = model(inputs)
                        loss = criterion(outputs, inputs)
                    elif model_type in ["vae", "beta_vae"]:
                        outputs, mu, logvar, _ = model(inputs)
                        beta = model.beta if hasattr(model, 'beta') else 1.0
                        loss = vae_loss(outputs, inputs, mu, logvar, beta)
                    elif model_type == "ensemble":
                        outputs, encodings = model(inputs)
                        loss = F.mse_loss(outputs, inputs)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # 메모리 정리
                    del inputs, loss
                    if model_type in ["vae", "beta_vae"]:
                        del outputs, mu, logvar
                    elif model_type == "ensemble":
                        del outputs, encodings
                    else:
                        del outputs
            
            val_loss = val_loss / val_batches if val_batches > 0 else 0.0

            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # TensorBoard logging
            writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

            # Print progress
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch:3d}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dim': input_dim,
                    'encoding_dim': encoding_dim,
                    'model_type': model_type,
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, model_save_path)
                joblib.dump(scaler, scaler_save_path)
                if epoch % 10 == 0 or epoch == 1:
                    print(f"✅ 모델 저장 - Val Loss: {val_loss:.6f}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"⏹️ Early stopping triggered at epoch {epoch}")
                    break
            
            # GPU 메모리 정리
            if torch.cuda.is_available() and epoch % 50 == 0:
                torch.cuda.empty_cache()

        writer.close()
        print("="*70)
        print(f"🎯 학습 완료!")
        print(f"📊 최고 검증 손실: {best_val_loss:.6f}")
        print(f"💾 모델 저장 경로: {model_save_path}")
        print(f"💾 Scaler 저장 경로: {scaler_save_path}")
        print("="*70)

        # 메모리 정리
        del model, optimizer, scheduler, X_train_tensor, X_test_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return None, scaler, model_save_path, scaler_save_path, log_dir

    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {str(e)}")
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e


def load_trained_model(model_path, scaler_path):
    """훈련된 모델 로드"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    input_dim = checkpoint['input_dim']
    encoding_dim = checkpoint['encoding_dim']
    model_type = checkpoint.get('model_type', 'improved')

    # 모델 타입에 따른 모델 생성
    if model_type == "improved":
        model = ImprovedAutoencoder(input_dim, encoding_dim).to(device)
    elif model_type == "vae":
        model = VariationalAutoencoder(input_dim, encoding_dim).to(device)
    elif model_type == "beta_vae":
        model = BetaVAE(input_dim, encoding_dim, beta=2.0).to(device)
    elif model_type == "ensemble":
        model = EnsembleAutoencoder(input_dim, encoding_dim, n_models=3).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = joblib.load(scaler_path)

    print(f"✅ 모델 로드 완료 - Type: {model_type}, Input dim: {input_dim}, Encoding dim: {encoding_dim}")
    print(f"✅ 최고 검증 손실: {checkpoint['best_val_loss']:.6f}")
    return model, scaler


def detect_anomalies(model, data_tensor, threshold_percentile=95):
    """이상치 탐지 함수"""
    model.eval()
    device = next(model.parameters()).device
    data_tensor = data_tensor.to(device)
    
    with torch.no_grad():
        if isinstance(model, (VariationalAutoencoder, BetaVAE)):
            reconstructed, mu, logvar, _ = model(data_tensor)
            # VAE의 경우 재구성 오차 + KL 발산 고려
            recon_error = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
            kl_error = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total_error = recon_error + 0.1 * kl_error  # KL 발산에 작은 가중치
        elif isinstance(model, EnsembleAutoencoder):
            reconstructed, _ = model(data_tensor)
            total_error = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
        else:
            reconstructed, _ = model(data_tensor)
            total_error = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
        
        threshold = torch.quantile(total_error, threshold_percentile / 100.0)
        anomalies = total_error > threshold
        
    return anomalies, total_error.cpu().numpy()