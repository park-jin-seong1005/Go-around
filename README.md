# Go-Around Prediction Project ✈️  

## 1. 프로젝트 개요  

### 프로젝트 명  
AI 기반 기상·항공 데이터 분석을 통한 Go-Around(복행) 예측 시스템  

### 참여 기간  
2024.XX ~ 2024.XX  

### 참가자  
박진성(팀장), [다른 팀원 이름]  

### 개발 목표  
- **제주공항 1년치 기상 데이터와 항공 데이터**를 분석해 Go-Around(복행) 발생 원인을 규명  
- 통계적 기법과 AI 모델을 결합해 **Go-Around 발생 확률 예측 시스템** 개발  
- OLLAMA 기반 챗봇과 연동해 **실시간 기상 데이터 기반 위험 분석** 제공  

---

## 2. 기술 스택  
- **데이터 분석**: Python, Pandas, NumPy, SciPy  
- **통계 분석**: Z-test (변수 검정)  
- **머신러닝 / AI**: Autoencoder, XGBoost  
- **시각화**: Matplotlib, Seaborn, HTML(지도 시각화)  
- **서비스 연계**: OLLAMA 기반 Chatbot  

---

## 3. 수행 작업  
- 제주공항 1년치 **기상 데이터 및 항공 데이터 수집·정제**  
- Z-test 활용하여 **Go-Around 발생에 영향을 주는 변수 분석**  
- 주요 변수들을 Autoencoder 및 XGBoost에 학습 → **발생 확률 예측 모델 구현**  
- **예측 파이프라인** 구축: 데이터 입력 → 전처리 → 모델 추론 → 위험도 산출  
- OLLAMA 기반 챗봇과 연동, 관제사/조종사가 **실시간 위험도 조회** 가능하도록 시스템 설계  
- 팀장으로서 프로젝트를 총괄하며 **데이터 기반 의사결정 및 모델 검증 과정** 주도  

---

## 4. 성과 / 배운 점  
- 데이터 정제 및 **AI 학습 파이프라인 구축 경험**  
- 통계 기법(Z-test) + AI 모델(자동인코더, XGBoost) **융합 분석 경험**  
- **실시간 예측 서비스**로 확장 가능한 구조 설계  
- 프로젝트 총괄 리더로서 **팀 협업 및 프로젝트 관리 능력 강화**  

---

## 5. 자료  

### 📂 Repository 구성  
- `Autoencoder/` : Autoencoder 모델 학습 코드 및 결과  
- `Captain/` : OLLAMA 기반 챗봇 및 시스템 연계 코드  
- `XGBoost.ipynb` : XGBoost 모델 학습 및 성능 평가  
- `Z_test.ipynb` : Z-test 통계 분석 코드  
- `fight_data.py` : 항공 데이터 처리 모듈  
- `weather_data.py` : 기상 데이터 처리 모듈  
- `histogram.ipynb` : 변수 분포 및 탐색적 데이터 분석(EDA)  
- `variable_heatmap.png` : 변수 상관관계 히트맵  
- `jeju_runway_area.html` : 제주공항 활주로 위치 시각화  

---

## 6. 참고 자료  
- [[DMS] Captain AI.pdf](https://github.com/user-attachments/files/22233671/DMS.Captain.AI.pdf)


---
