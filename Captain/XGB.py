import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

input_path = r"/home/kng/GO-AROUND/csv/5904개중 59개인데 고어라운드 안된거(valid).csv"
output_path = r"./output/output1.csv"

# CSV 읽기
df = pd.read_csv(input_path)

# 로그 변환 (음수 값 포함 변환)
df_log = df.copy()

def signed_log_transform(x):
    if pd.isna(x):
        return np.nan
    if x > 0:
        return np.log(x + 1)
    elif x < 0:
        return -np.log(abs(x) + 1)
    else:
        return 0

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df_log[col] = df[col].apply(signed_log_transform)

# 변환 결과 저장
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_log.to_csv(output_path, index=False)

# 스케일러 & 모델 로드
scaler = joblib.load('/home/kng/GO-AROUND/scaler.pkl')
model = joblib.load('./xgb_model_cweight1.1.pkl')

# 변환된 CSV 불러오기
output1 = pd.read_csv(output_path)

# 정규화
output1_scaled = scaler.transform(output1)

# 예측 확률 및 클래스
realproba = model.predict_proba(output1_scaled)[:, 1]
predicted_class = (realproba >= 0.5).astype(int)

# 결과 DataFrame에 추가
output1['goaround_prob'] = realproba
output1['goaround_pred'] = predicted_class

# 확률(%) 변환
realproba_percent = realproba * 100

# 행별 확률 출력
print("\n=== 개별 행 예측 확률 (%) ===")
for idx, prob in enumerate(realproba_percent, start=1):
    print(f"Row {idx}: {prob:.6f}%")

# 표 형태 출력
print("\n=== 확률 표 ===")
prob_df = pd.DataFrame({
    'Row': range(1, len(realproba_percent) + 1),
    'Go-Around Probability (%)': realproba_percent
})
print(prob_df)

# 결과 저장
output_with_preds_path = r"./output/output_with_predictions.csv"
output1.to_csv(output_with_preds_path, index=False)
print(f"\n결과 저장 완료: {output_with_preds_path}")



# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import os


# input_path = r"./output/output.csv"
# output_path = r"./output/output1.csv"

# # CSV 읽기
# df = pd.read_csv(input_path)

# # 로그 변환 (음수 값 포함 변환)
# df_log = df.copy()

# def signed_log_transform(x):
#     if pd.isna(x):
#         return np.nan
#     if x > 0:
#         return np.log(x + 1)
#     elif x < 0:
#         return -np.log(abs(x) + 1)
#     else:
#         return 0

# for col in df.columns:
#     if pd.api.types.is_numeric_dtype(df[col]):
#         df_log[col] = df[col].apply(signed_log_transform)

# # 저장
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# df_log.to_csv(output_path, index=False)


# scaler = joblib.load('/home/kng/GO-AROUND/scaler.pkl')

# model = joblib.load('./xgb_model_cweight1.1.pkl')

# # 1. 원본 데이터 불러오기
# output1 = pd.read_csv('./output/output1.csv')

# # 3. 정규화
# output1_scaled = scaler.transform(output1)

# # 4. 예측 확률 및 클래스
# realproba = model.predict_proba(output1_scaled)[:, 1]
# predicted_class = (realproba >= 0.5).astype(int)

# # 5. 결과 붙이기 및 저장
# output1['goaround_prob'] = realproba
# output1['goaround_pred'] = predicted_class

# print(f'{realproba*100}%')


