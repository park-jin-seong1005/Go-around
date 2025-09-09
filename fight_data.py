
import requests
import pandas as pd
import os
import time
from datetime import datetime, timezone, timedelta
import json

# 🔹 API 토큰
API_TOKEN = "9e758b51-a861-4f8d-b81d-5912d0f9f598|8tGLwhDDO6KCkgJLQBO1Xhf8FK7bWqJDZrOM2gd29e166f9d"
BASE_URL = "https://fr24api.flightradar24.com/api"

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept-Version": "v1"
}


# UTC to KST 변환 함수
def convert_utc_to_kst(utc_timestamp):
    """UTC 시간을 KST로 변환하는 함수"""
    utc_time = datetime.strptime(utc_timestamp, "%Y-%m-%dT%H:%M:%SZ")
    kst_time = utc_time + timedelta(hours=9)  # KST는 UTC보다 9시간 앞
    return kst_time


def load_existing_flight_ids(base_folder):
    processed = set()
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".csv"):
                # 파일명에서 flight_id 추출 (예: abcd1234_14h.csv → abcd1234)
                flight_id = file.split("_")[0]
                processed.add(flight_id)
    return processed


def get_flights_to_jeju_by_hour(date, hour):
    # 날짜와 시간 조합을 UTC 타임스탬프로 변환
    dt = datetime.strptime(f"{date} {hour:02d}:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    timestamp = int(dt.timestamp())

    url = f"{BASE_URL}/historic/flight-positions/full"
    params = {
        "timestamp": timestamp,
        "airports": "inbound:CJU",
        "limit": 1000
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # 응답 로그 저장
        os.makedirs("api_logs", exist_ok=True)
        with open(f"api_logs/{date}_{hour:02d}h.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return data.get("data", []) if isinstance(data, dict) else data
    except Exception as e:
        print(f"❌ {date} {hour:02d}h 오류: {e}")
        return []


def get_flight_track(flight_id):
    url = f"{BASE_URL}/flight-tracks"
    params = {"flight_id": flight_id}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("data", data) if isinstance(data, dict) else data
    except Exception as e:
        print(f"❌ 항공편 {flight_id} 트랙 오류: {e}")
        return []


def save_to_csv(flight_id, date, hour, data, base_folder):
    folder_name = os.path.join(base_folder, f"{date}")
    os.makedirs(folder_name, exist_ok=True)

    file_path = os.path.join(folder_name, f"{flight_id}_{hour:02d}h.csv")

    try:
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                df = pd.DataFrame(data)
            elif isinstance(data[0], list):
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'lat', 'lon', 'alt', 'gspeed', 'vspeed', 'track', 'squawk', 'callsign', 'source'])
            else:
                df = pd.DataFrame({'data': [str(data)]})
        else:
            df = pd.DataFrame({'data': [str(data)]})

        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"📁 저장 완료: {file_path}")
        return True
    except Exception as e:
        print(f"❌ 파일 저장 오류 ({file_path}): {e}")
        return False


def process_date(date, processed_flights, base_folder):
    """특정 날짜의 모든 시간대에 대해 제주공항 착륙 비행기 데이터를 수집"""
    print(f"📅 === {date} 데이터 수집 시작 === 📅")

    # 성공적으로 처리된 항공편 수
    successful_flights = 0

    # 해당 날짜의 모든 시간대(0-23시)에 대해 처리
    for hour in range(24):
        print(f"🕒 {date} {hour:02d}h 항공편 검색 중...")
        flights = get_flights_to_jeju_by_hour(date, hour)

        if not flights:
            print(f"ℹ️ {date} {hour:02d}h 항공편 정보 없음")
            continue

        print(f"📊 {date} {hour:02d}h 항공편 {len(flights)}개 발견")

        for flight in flights:
            flight_id = flight.get("fr24_id")
            if not flight_id:
                continue

            if flight_id in processed_flights:
                print(f"🔄 이미 처리된 항공편 건너뜀: {flight_id}")
                continue

            print(f"✈️ 트랙 조회: {flight_id} ({date} {hour:02d}h)")

            track_data = get_flight_track(flight_id)
            if not track_data:
                print(f"⚠️ {flight_id} 트랙 데이터 없음")
                continue

            # 항공편과 날짜 일치 확인
            if isinstance(track_data, list) and track_data and 'timestamp' in track_data[0]:
                first_timestamp = track_data[0]['timestamp']
                flight_date = first_timestamp.split('T')[0]

                if flight_date != date:
                    print(f"⚠️ 날짜 불일치: 요청 날짜 {date}, 실제 항공편 날짜 {flight_date}, 건너뜀")
                    continue

            # 저장 시도
            save_success = save_to_csv(flight_id, date, hour, track_data, base_folder)

            if save_success:
                processed_flights.add(flight_id)
                successful_flights += 1

            # API 요청 간 간격 두기
            time.sleep(2)

    print(f"✅ === {date} 데이터 수집 완료 ({successful_flights}개 항공편 저장) === ✅")
    # 날짜 간 처리 간격 - API 제한 고려
    time.sleep(5)
    return processed_flights


def collect_data_for_dates(dates_file):
    """텍스트 파일에서 날짜 목록을 읽어와 순차적으로 처리"""
    # 날짜 목록 읽기
    with open(dates_file, 'r', encoding='utf-8') as file:
        dates = [line.strip() for line in file if line.strip()]

    # 날짜 형식 확인 및 정렬
    valid_dates = []
    for date in dates:
        try:
            datetime.strptime(date, "%Y-%m-%d")
            valid_dates.append(date)
        except ValueError:
            print(f"⚠️ 잘못된 날짜 형식 무시: {date}")

    # 날짜순으로 정렬
    valid_dates.sort()

    print(f"🗓️ 총 {len(valid_dates)}개 유효 날짜 처리 예정")

    # 이미 처리된 날짜 확인
    processed_dates_file = "processed_dates.txt"
    processed_dates = set()
    if os.path.exists(processed_dates_file):
        with open(processed_dates_file, 'r', encoding='utf-8') as file:
            processed_dates = {line.strip() for line in file if line.strip()}

    # 처리할 날짜 필터링
    dates_to_process = [date for date in valid_dates if date not in processed_dates]
    print(f"🔄 이미 처리된 날짜: {len(processed_dates)}개, 처리할 날짜: {len(dates_to_process)}개")

    if not dates_to_process:
        print("✅ 모든 날짜가 이미 처리되었습니다!")
        return

    # 데이터 저장 기본 폴더 설정
    base_folder = r"C:\Users\USER\Desktop\FLIGHT_DATA\Go_Around\Jeju_Flights"
    os.makedirs(base_folder, exist_ok=True)

    # 기존에 처리한 flight_id 로드
    processed_flights = load_existing_flight_ids(base_folder)
    print(f"🔄 기존 처리된 항공편: {len(processed_flights)}개")

    # 날짜별로 순차 처리
    for date in dates_to_process:
        try:
            processed_flights = process_date(date, processed_flights, base_folder)

            # 진행상황 저장 - 날짜마다 완료된 날짜 기록
            with open(processed_dates_file, "a", encoding='utf-8') as f:
                f.write(f"{date}\n")

        except Exception as e:
            print(f"❌ {date} 처리 중 오류 발생: {e}")
            # 오류 발생 시에도 진행 중단하지 않고 기록
            with open("error_dates.txt", "a", encoding='utf-8') as f:
                f.write(f"{date}: {str(e)}\n")

    print("🎉 모든 날짜 처리 완료!")


if __name__ == "__main__":
    dates_file = r"C:\Users\USER\PycharmProjects\PythonProject1\paste.txt.txt"  # 날짜 목록이 있는 텍스트 파일 경로
    collect_data_for_dates(dates_file)