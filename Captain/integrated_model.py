import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================================
# 데이터 전처리 모듈
# ================================

def signed_log_transform(x):
    """음수 값을 포함한 로그 변환"""
    if pd.isna(x):
        return np.nan
    if x > 0:
        return np.log(x + 1)
    elif x < 0:
        return -np.log(abs(x) + 1)
    else:
        return 0

def preprocess_data(input_path):
    """데이터 로그 변환 (메모리에서만 처리)"""
    df = pd.read_csv(input_path)
    df_log = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df_log[col] = df[col].apply(signed_log_transform)
    
    return df_log

# ================================
# 오토인코더 모델 클래스들
# ================================

class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=4):
        super(ImprovedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU()
        )

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

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

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

class BetaVAE(VariationalAutoencoder):
    def __init__(self, input_dim, latent_dim=4, beta=1.0):
        super(BetaVAE, self).__init__(input_dim, latent_dim)
        self.beta = beta

class EnsembleAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=3, n_models=3):
        super(EnsembleAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.n_models = n_models
        
        self.autoencoders = nn.ModuleList()
        
        for i in range(n_models):
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
            
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output, encodings

# ================================
# 오토인코더 예측 모듈
# ================================

def load_trained_model(model_path, scaler_path):
    """훈련된 모델 로드"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    input_dim = checkpoint['input_dim']
    encoding_dim = checkpoint['encoding_dim']
    model_type = checkpoint.get('model_type', 'improved')

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
    return model, scaler

def predict_autoencoder_probability(model_path, scaler_path, test_data, normal_data_path):
    """오토인코더 기반 Go-around 확률 예측"""
    # 모델과 스케일러 로드
    model, scaler = load_trained_model(model_path, scaler_path)
    
    # 테스트 데이터 처리
    test_df = test_data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # 정상 데이터 로드 (베이스라인 계산용)
    normal_df = pd.read_csv(normal_data_path, encoding='utf-8-sig')
    normal_df = normal_df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # 데이터 전처리
    test_scaled = scaler.transform(test_df)
    normal_scaled = scaler.transform(normal_df)
    
    # 텐서로 변환
    device = next(model.parameters()).device
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32).to(device)
    normal_tensor = torch.tensor(normal_scaled, dtype=torch.float32).to(device)
    
    # 모델 평가 모드
    model.eval()
    
    with torch.no_grad():
        # 테스트 데이터 재구성 오차 계산
        if isinstance(model, (VariationalAutoencoder, BetaVAE)):
            test_reconstructed, mu, logvar, _ = model(test_tensor)
            test_recon_error = torch.mean((test_tensor - test_reconstructed) ** 2, dim=1)
        elif isinstance(model, EnsembleAutoencoder):
            test_reconstructed, _ = model(test_tensor)
            test_recon_error = torch.mean((test_tensor - test_reconstructed) ** 2, dim=1)
        else:
            test_reconstructed, _ = model(test_tensor)
            test_recon_error = torch.mean((test_tensor - test_reconstructed) ** 2, dim=1)
        
        # 정상 데이터 재구성 오차 계산
        if isinstance(model, (VariationalAutoencoder, BetaVAE)):
            normal_reconstructed, mu_normal, logvar_normal, _ = model(normal_tensor)
            normal_recon_error = torch.mean((normal_tensor - normal_reconstructed) ** 2, dim=1)
        elif isinstance(model, EnsembleAutoencoder):
            normal_reconstructed, _ = model(normal_tensor)
            normal_recon_error = torch.mean((normal_tensor - normal_reconstructed) ** 2, dim=1)
        else:
            normal_reconstructed, _ = model(normal_tensor)
            normal_recon_error = torch.mean((normal_tensor - normal_reconstructed) ** 2, dim=1)
    
    # CPU로 이동
    test_errors = test_recon_error.cpu().numpy()
    normal_errors = normal_recon_error.cpu().numpy()
    
    # 정상 데이터 통계량 계산
    normal_mean = np.mean(normal_errors)
    normal_std = np.std(normal_errors)
    
    # 확률 계산
    probabilities = []
    for test_error in test_errors:
        z_score = (test_error - normal_mean) / normal_std if normal_std > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)
        go_around_prob = (1 - p_value) * 100
        probabilities.append(go_around_prob)
    
    return np.array(probabilities)

# ================================
# XGBoost 예측 모듈
# ================================

def predict_xgboost_probability(processed_data, scaler_path, xgb_model_path):
    """XGBoost 기반 Go-around 확률 예측"""
    # 모델과 스케일러 로드
    scaler = joblib.load(scaler_path)
    model = joblib.load(xgb_model_path)
    
    # 정규화
    data_scaled = scaler.transform(processed_data)
    
    # 예측 확률 계산
    probabilities = model.predict_proba(data_scaled)[:, 1] * 100
    
    return probabilities

# ================================
# 통합 예측 시스템
# ================================

class GoAroundPredictor:
    def __init__(self):
        self.autoencoder_prob = None
        self.xgboost_prob = None
        self.ensemble_prob = None
        self.processed_data = None
    
    def predict(self, 
                input_data_path,
                # 오토인코더 관련 파라미터
                autoencoder_model_path,
                autoencoder_scaler_path, 
                normal_data_path,
                # XGBoost 관련 파라미터
                xgb_model_path,
                xgb_scaler_path):
        
        # 1. 데이터 전처리 (메모리에서만)
        self.processed_data = preprocess_data(input_data_path)
        
        # 2. 오토인코더 예측
        try:
            self.autoencoder_prob = predict_autoencoder_probability(
                autoencoder_model_path, 
                autoencoder_scaler_path, 
                self.processed_data, 
                normal_data_path
            )
            #print(f"오토인코더 예측 완료: 평균 확률 {np.mean(self.autoencoder_prob):.2f}%")
        except Exception as e:
            print(f"오토인코더 예측 실패: {e}")
            self.autoencoder_prob = None
        
        # 3. XGBoost 예측
        try:
            self.xgboost_prob = predict_xgboost_probability(
                self.processed_data, 
                xgb_scaler_path, 
                xgb_model_path
            )
            #print(f"XGBoost 예측 완료: 평균 확률 {np.mean(self.xgboost_prob):.2f}%")
        except Exception as e:
            print(f"XGBoost 예측 실패: {e}")
            self.xgboost_prob = None
        
        # 4. 앙상블 예측 (평균)
        if self.autoencoder_prob is not None and self.xgboost_prob is not None:
            #print(f'{self.autoencoder_prob},{self.xgboost_prob}')
            self.ensemble_prob = (self.autoencoder_prob*0.3 + self.xgboost_prob*0.7)
            #print(f"앙상블 예측 완료: 평균 확률 {np.mean(self.ensemble_prob):.2f}%")
        elif self.autoencoder_prob is not None:
            self.ensemble_prob = self.autoencoder_prob
            #print("오토인코더 예측만 사용")
        elif self.xgboost_prob is not None:
            self.ensemble_prob = self.xgboost_prob
            #print("XGBoost 예측만 사용")
        else:
            raise Exception("모든 예측 모델이 실패했습니다.")
        
        return {
            'autoencoder_prob': self.autoencoder_prob,
            'xgboost_prob': self.xgboost_prob,
            'ensemble_prob': self.ensemble_prob,
            'processed_data': self.processed_data
        }
    
    def get_results_dataframe(self):
        """결과를 DataFrame으로 반환"""
        if self.processed_data is None:
            raise ValueError("예측을 먼저 실행해주세요.")
        
        result_df = self.processed_data.copy()
        
        # 결과 추가
        if self.autoencoder_prob is not None:
            result_df['autoencoder_prob'] = self.autoencoder_prob
        if self.xgboost_prob is not None:
            result_df['xgboost_prob'] = self.xgboost_prob
        if self.ensemble_prob is not None:
            result_df['ensemble_prob'] = self.ensemble_prob
            result_df['ensemble_pred'] = (self.ensemble_prob >= 50).astype(int)
        
        return result_df

# ================================
# 사용 예시
# ================================

if __name__ == "__main__":
    # 파라미터 설정
    INPUT_DATA_PATH = "./output/output.csv"
    
    # 오토인코더 관련 파라미터
    AUTOENCODER_MODEL_PATH = '/home/kng/Captain/new_autoencoder/best_autoencoder.pt'
    AUTOENCODER_SCALER_PATH = '/home/kng/Captain/new_autoencoder/scaler.pkl'
    NORMAL_DATA_PATH = '/home/kng/GO-AROUND/csv/59개59개 빼고 남은거(학습용데이터)(로그변환).csv'
    
    # XGBoost 관련 파라미터
    XGB_MODEL_PATH = './xgb_model_cweight1.1.pkl'
    XGB_SCALER_PATH = '/home/kng/GO-AROUND/scaler.pkl'
    
    # 예측 실행
    predictor = GoAroundPredictor()
    
    try:
        results = predictor.predict(
            input_data_path=INPUT_DATA_PATH,
            autoencoder_model_path=AUTOENCODER_MODEL_PATH,
            autoencoder_scaler_path=AUTOENCODER_SCALER_PATH,
            normal_data_path=NORMAL_DATA_PATH,
            xgb_model_path=XGB_MODEL_PATH,
            xgb_scaler_path=XGB_SCALER_PATH
        )
        
        #print("\n=== 최종 결과 ===")
        if results['ensemble_prob'] is not None:
            print(f"평균 확률: {np.mean(results['ensemble_prob']):.2f}%")
        
        # 결과 DataFrame 가져오기 (필요시)
        # result_df = predictor.get_results_dataframe()
        # print(result_df.head())
            
    except Exception as e:
        print(f"예측 실패: {e}")