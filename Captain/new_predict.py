
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 오토인코더 모델 클래스들 (paste.txt에서 가져온 것)
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

    # print(f"✅ 모델 로드 완료 - Type: {model_type}, Input dim: {input_dim}, Encoding dim: {encoding_dim}")
    # print(f"✅ 최고 검증 손실: {checkpoint['best_val_loss']:.6f}")
    return model, scaler


def detect_go_around_probability_statistical(model_path, test_data_path, scaler_path, normal_data_path, probability_method='one_tailed'):
    """Go-around 확률 통계적 분석"""
    # 모델과 스케일러 로드
    model, scaler = load_trained_model(model_path, scaler_path)
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(test_data_path, encoding='utf-8-sig')
    test_df = test_df.apply(pd.to_numeric, errors='coerce').dropna()
    
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
        
        # 정상 데이터 재구성 오차 계산 (베이스라인)
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
    
    # 각 테스트 샘플에 대한 분석
    results = []
    
    for i, test_error in enumerate(test_errors):
        # Z-score 계산
        z_score = (test_error - normal_mean) / normal_std if normal_std > 0 else 0
        
        # P-value 계산
        if probability_method == 'one_tailed':
            p_value = 1 - stats.norm.cdf(z_score)  # 오른쪽 꼬리 (높은 오차가 이상)
        else:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # 양쪽 꼬리
        
        # Go-around 확률 (백분율)
        go_around_prob_percent = (1 - p_value) * 100
        
        # 리스크 레벨
        if p_value < 0.05:
            risk_level = "HIGH"
        elif p_value < 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        results.append({
            'sample_index': i,
            'reconstruction_error': test_error,
            'z_score': z_score,
            'p_value': p_value,
            'go_around_probability_percent': go_around_prob_percent,
            'risk_level': risk_level
        })
    
    # 결과를 DataFrame으로 변환
    row_results = pd.DataFrame(results)
    
    # 원본 데이터와 결과 합치기
    df_original = test_df.copy()
    df_combined = pd.concat([df_original.reset_index(drop=True), row_results.reset_index(drop=True)], axis=1)
    
    return df_combined, row_results


# 단일 모델 Go-Around 확률 분석 코드
import os
import pandas as pd
from tqdm import tqdm

def analyze_single_model_go_around(model_path, scaler_path, test_data_path, normal_data_path, output_dir=None):

    
    # 파일 존재 여부 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"테스트 데이터 파일을 찾을 수 없습니다: {test_data_path}")
    if not os.path.exists(normal_data_path):
        raise FileNotFoundError(f"정상 데이터 파일을 찾을 수 없습니다: {normal_data_path}")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    # 분석 결과 저장할 디렉토리 생성
    analysis_output_dir = os.path.join(output_dir, "go_around_analysis_statistical")
    os.makedirs(analysis_output_dir, exist_ok=True)

    
    try:
        # Go-Around 확률 탐지 실행
        df_original, row_results = detect_go_around_probability_statistical(
            model_path=model_path,
            test_data_path=test_data_path,
            scaler_path=scaler_path,
            normal_data_path=normal_data_path,
            probability_method='one_tailed'
        )
        
        # 모델명 (파일명에서 확장자 제거)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # 개별 행 결과 CSV 저장
        detailed_csv_path = os.path.join(analysis_output_dir, f"{model_name}_detailed_results.csv")
        row_results.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
        
        # 평균 통계 계산
        avg_prob_percent = row_results['go_around_probability_percent'].mean()
        avg_reconstruction_error = row_results['reconstruction_error'].mean()
        avg_z_score = row_results['z_score'].mean()
        avg_p_value = row_results['p_value'].mean()
        
        # 리스크 레벨 결정
        if avg_p_value < 0.05:
            risk_level = "HIGH"
        elif avg_p_value < 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # 요약 결과 생성
        summary_result = {
            'model_name': model_name,
            'model_path': model_path,
            'total_test_samples': len(row_results),
            'avg_reconstruction_error': avg_reconstruction_error,
            'avg_z_score': avg_z_score,
            'avg_p_value': avg_p_value,
            'avg_go_around_probability_percent': avg_prob_percent,
            'risk_level': risk_level,
            'high_risk_samples': len(row_results[row_results['p_value'] < 0.05]),
            'medium_risk_samples': len(row_results[(row_results['p_value'] >= 0.05) & (row_results['p_value'] < 0.1)]),
            'low_risk_samples': len(row_results[row_results['p_value'] >= 0.1])
        }
        
        # 요약 결과 DataFrame 생성 및 CSV 저장
        summary_df = pd.DataFrame([summary_result])
        summary_csv_path = os.path.join(analysis_output_dir, f"{model_name}_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        
        # 결과 출력

        
        print(avg_prob_percent)
        # 완료 마커 생성
        done_marker = os.path.join(analysis_output_dir, "done.marker")
        with open(done_marker, 'w') as f:
            f.write("analysis_completed")
        
        return row_results, summary_result
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        raise


# 사용 예시
if __name__ == "__main__":
    # 경로 설정
    MODEL_PATH = r"C:\Users\USER\Desktop\FLIGHT_DATA\workspace\autoencoder\newmodel\runs2\runs\beta_vae_bs4_lr0.0001_ep500_20250713-205435\best_autoencoder.pt"
    SCALER_PATH = r"C:\Users\USER\Desktop\FLIGHT_DATA\workspace\autoencoder\newmodel\runs2\runs\beta_vae_bs4_lr0.0001_ep500_20250713-205435\scaler.pkl"
    TEST_DATA_PATH = r"C:\Users\USER\Desktop\FLIGHT_DATA\workspace\autoencoder\고어라운드_된_날씨_합친_59개_로그변환(test).csv"
    NORMAL_DATA_PATH = r"C:\Users\USER\Desktop\FLIGHT_DATA\workspace\autoencoder\59개59개 빼고 남은거(학습용데이터)(로그변환).csv"
    
    # 단일 모델 분석 실행
    try:
        detailed_results, summary = analyze_single_model_go_around(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            test_data_path=TEST_DATA_PATH,
            normal_data_path=NORMAL_DATA_PATH
        )
                
    except Exception as e:
        print(f"❌ 오류 발생: {e}")