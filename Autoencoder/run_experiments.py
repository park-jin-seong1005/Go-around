import itertools
import pandas as pd
import time
import os
import sys
import threading
from datetime import datetime
from autoencoder_newmodule import train_autoencoder

# 전역 락으로 중복 실행 방지
experiment_lock = threading.Lock()

def main():
    print("🔥 실험 시작 시간:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    # runs 폴더 경로 설정 및 생성
    base_runs_path = r"C:\Users\USER\Desktop\FLIGHT_DATA\workspace\autoencoder\newmodel\runs2"
    if not os.path.exists(base_runs_path):
        os.makedirs(base_runs_path)
        print(f"📁 runs 폴더 생성: {base_runs_path}")
    else:
        print(f"📁 runs 폴더 확인: {base_runs_path}")
    
    # 실험할 하이퍼파라미터 조합들

    model_types = ["improved", "vae", "beta_vae", "ensemble"]  # 새로운 모델 타입들
    batch_sizes = [4, 8, 16, 32, 64,]
    learning_rates = [
            # 높은 학습률 (빠른 학습)
            1e-2, 5e-3, 3e-3, 2e-3, 1e-3,
            # 중간 학습률 (균형)
            9e-4, 7e-4, 5e-4, 3e-4, 2e-4, 1e-4,
            # 낮은 학습률 (정밀한 학습)
            9e-5, 7e-5, 5e-5, 3e-5, 2e-5, 1e-5,
        ]    
    epochs_list = [500]
    
    # 로그 변환된 데이터 경로
    data_path = r"C:\Users\USER\Desktop\FLIGHT_DATA\workspace\autoencoder\59개59개 빼고 남은거(학습용데이터)(로그변환).csv"
    
    # 데이터 파일 존재 확인
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        sys.exit(1)
    
    # 결과 저장용 리스트와 파일명 설정 (runs 폴더 안에 저장)
    results = []
    experiment_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join(base_runs_path, f"experiment_results_{experiment_start_time}.csv")
    
    print(f"📝 결과는 '{results_filename}' 파일에 저장됩니다.")
    
    # 전체 실험 개수 계산
    total_experiments = len(model_types) * len(batch_sizes) * len(learning_rates) * len(epochs_list)
    print(f"📊 총 {total_experiments}개의 실험을 진행합니다.")
    print(f"🤖 Model types: {len(model_types)}개 - {model_types}")
    print(f"🔢 Batch sizes: {len(batch_sizes)}개 - {batch_sizes}")
    print(f"🔢 Learning rates: {len(learning_rates)}개 - 범위: {min(learning_rates):.0e} ~ {max(learning_rates):.0e}")
    print(f"🔢 Epochs: {epochs_list}")
    print(f"⏱️ 예상 소요시간: 약 {total_experiments * 4:.0f}분 ({total_experiments * 4 / 60:.1f}시간)")
    print("="*80)
    
    # 하이퍼파라미터 조합 루프
    experiment_count = 0
    
    for model_type, batch_size, lr, epochs in itertools.product(model_types, batch_sizes, learning_rates, epochs_list):
        experiment_count += 1
        
        print(f"\n🚧 실험 {experiment_count}/{total_experiments}")
        print(f"📝 설정 - Batch Size: {batch_size}, LR: {lr}, Epochs: {epochs}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            print(f"⏰ 실험 {experiment_count} 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
            
            # 현재 작업 디렉토리를 runs 폴더로 임시 변경
            original_cwd = os.getcwd()
            os.chdir(base_runs_path)
            
            try:
                # 실험 실행
                model, scaler, model_path, scaler_path, log_dir = train_autoencoder(
                    data_path=data_path,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    model_type=model_type
                )
                
                # 상대 경로를 절대 경로로 변환
                if model_path and not os.path.isabs(model_path):
                    model_path = os.path.join(base_runs_path, model_path)
                if scaler_path and not os.path.isabs(scaler_path):
                    scaler_path = os.path.join(base_runs_path, scaler_path)
                if log_dir and not os.path.isabs(log_dir):
                    log_dir = os.path.join(base_runs_path, log_dir)
                    
            finally:
                # 원래 작업 디렉토리로 복원
                os.chdir(original_cwd)
            
            duration = time.time() - start_time
            print(f"⏰ 실험 {experiment_count} 완료 시간: {datetime.now().strftime('%H:%M:%S')}")
            
            # 결과 기록
            results.append({
                "experiment_id": experiment_count,
                "model_type": model_type,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "model_path": model_path,
                "scaler_path": scaler_path,
                "log_folder": log_dir,
                "train_time_sec": round(duration, 2),
                "status": "completed",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"✅ 실험 {experiment_count}/{total_experiments} 완료!")
            print(f"   🤖 모델타입: {model_type}")
            print(f"   📊 배치크기: {batch_size}, 학습률: {lr}")
            print(f"   ⏱️ 소요시간: {duration:.2f}초 ({duration/60:.1f}분)")
            print(f"   💾 모델저장: {model_path}")
            print(f"   💾 스케일러: {scaler_path}")
            print(f"   📊 로그폴더: {log_dir}")
            
            
            # 실험 완료 후 즉시 CSV에 기록
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_filename, index=False)
            print(f"   📝 결과 업데이트: {os.path.basename(results_filename)} (총 {len(results)}개 실험 완료)")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 실험 {experiment_count} 실패: {str(e)}")
            
            # 실패한 실험도 기록
            results.append({
                "experiment_id": experiment_count,
                "model_type": model_type,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "model_path": None,
                "scaler_path": None,
                "log_folder": None,
                "train_time_sec": None,
                "train_time_min": None,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 실패한 실험도 즉시 CSV에 기록
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_filename, index=False)
            print(f"   📝 실패 결과도 업데이트: {os.path.basename(results_filename)}")
            
            # 에러 발생 시에도 계속 진행
            continue  # 다음 실험 계속 진행
    
    # 최종 요약 출력
    print("\n" + "="*80)
    print("🎯 모든 개선된 오토인코더 실험 완료!")
    print(f"📊 총 {len(results)}개 실험 수행")
    print(f"✅ 성공: {len([r for r in results if r['status'] == 'completed'])}개")
    print(f"❌ 실패: {len([r for r in results if r['status'] == 'failed'])}개")
    print(f"💾 최종 결과: {results_filename}")
    print(f"📁 모든 파일 저장 위치: {base_runs_path}")
    print("="*80)

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때만 main 함수 호출
    main()