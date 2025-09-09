import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import time
import csv
import os
import json


def fetch_amos_data_by_hour(date_str, hour_str):
    """
    기상청 AMOS API에서 특정 날짜의 특정 시간대 데이터를 1분 단위로 가져오는 함수
    - date_str: 'YYYYMMDD' 형식의 날짜 문자열
    - hour_str: 'HH' 형식의 시간 문자열 (00~23)
    """
    # API 설정
    auth_key = 'vbLwQ6jMTmuy8EOozJ5rew'
    base_url = 'https://apihub.kma.go.kr/api/typ01/url/amos.php?tm=202211301200&dtm=60&help=1&authKey=vbLwQ6jMTmuy8EOozJ5rew'

    # 데이터 요청 (1분 단위)
    params = {
        'tm': f'{date_str}{hour_str}00',  # 시작 시간 (해당 날짜 HH:00)
        'dtm': '60',  # 데이터 간격 (60초 = 1분)
        'stn': '182',  # 모든 관측소
        'help': '1',  # 도움말 포함
        'authKey': auth_key  # 인증 키
    }

    request_url = f"{base_url}?tm={params['tm']}&dtm={params['dtm']}&stn={params['stn']}&help={params['help']}&authKey={params['authKey']}"
    print(f"요청 URL: {request_url}")

    try:
        # API 요청
        response = requests.get(base_url, params=params)

        # 응답 확인
        if response.status_code != 200:
            print(f"API 요청 실패. 상태 코드: {response.status_code}")
            return None

        # 응답 본문 가져오기
        content = response.text
        print(f"응답 받음. 길이: {len(content)} 바이트")

        # 데이터 검사
        if len(content.strip()) < 10:  # 너무 짧은 응답은 무의미할 가능성 높음
            print("응답이 너무 짧습니다. 유효한 데이터가 아닐 수 있습니다.")
            return None

        return content

    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def collect_day_data(output_path, date_str, temp_dir=None):
    """
    특정 날짜의 0시부터 23시까지 시간별로 데이터를 수집하여 하나의 txt 파일로 저장
    """
    # 임시 저장할 내용
    combined_content = ""

    # 성공 및 실패 카운트
    success_count = 0
    failed_hours = []

    # 수집 시작
    print(f"\n{date_str} 데이터 수집 시작 (0시~23시, 1분 단위)")

    # 0시부터 23시까지 반복
    for hour in range(24):
        hour_str = f"{hour:02d}"

        print(f"{date_str} {hour_str}시 데이터 수집 중...")

        # 데이터 요청
        content = fetch_amos_data_by_hour(date_str, hour_str)

        # 데이터 저장
        if content and len(content.strip()) > 10:
            # 시간 구분자 추가
            combined_content += f"\n\n===== {date_str} {hour_str}:00 =====\n\n"
            combined_content += content
            success_count += 1
        else:
            print(f"{date_str} {hour_str}시: 데이터 수집 실패")
            failed_hours.append(hour_str)

        # API 과부하 방지를 위한 지연
        time.sleep(1)

    # 실패한 시간대가 있다면 재시도 정보 저장
    if failed_hours:
        if temp_dir is not None:
            retry_info = {
                'date': date_str,
                'failed_hours': failed_hours
            }
            retry_file = os.path.join(temp_dir, f'retry_{date_str}.json')
            with open(retry_file, 'w') as f:
                json.dump(retry_info, f)

    # 데이터 저장
    if combined_content:
        # 출력 디렉토리 확인 및 생성
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 파일로 저장
        file_path = os.path.join(output_path, f'amos_data_{date_str}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        print(f"{date_str}: 데이터를 {file_path} 파일로 저장했습니다.")

        # 결과 출력
        print(f"{date_str}: 성공 {success_count}시간, 실패 {len(failed_hours)}시간")

        return True, failed_hours
    else:
        print(f"{date_str}: 저장할 데이터가 없습니다.")
        return False, failed_hours


def retry_failed_days(output_path, failed_days, temp_dir):
    """실패한 날짜 재시도"""
    print("\n\n재시도: 실패한 날짜 처리 중...")

    still_failed = []

    for date_str in failed_days:
        print(f"\n재시도: {date_str} 데이터 수집 중...")
        success, _ = collect_day_data(output_path, date_str)

        if not success:
            still_failed.append(date_str)

    return still_failed


def retry_failed_hours(output_path, temp_dir):
    """실패한 시간대 재시도"""
    print("\n\n재시도: 실패한 시간대 처리 중...")

    # 재시도 파일 목록
    retry_files = [f for f in os.listdir(temp_dir) if f.startswith('retry_') and f.endswith('.json')]

    if not retry_files:
        print("재시도할 항목이 없습니다.")
        return

    for retry_file in retry_files:
        file_path = os.path.join(temp_dir, retry_file)

        with open(file_path, 'r') as f:
            retry_info = json.load(f)

        date_str = retry_info['date']
        failed_hours = retry_info['failed_hours']

        print(f"\n재시도: {date_str}의 실패한 시간대 ({','.join(failed_hours)}) 처리 중...")

        # 기존 파일 읽기
        existing_file = os.path.join(output_path, f'amos_data_{date_str}.txt')
        existing_content = ""

        if os.path.exists(existing_file):
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()

        # 실패한 시간대 재시도
        for hour_str in failed_hours:
            print(f"{date_str} {hour_str}시 데이터 재수집 중...")

            # 데이터 요청
            content = fetch_amos_data_by_hour(date_str, hour_str)

            # 데이터 추가
            if content and len(content.strip()) > 10:
                # 시간 구분자 추가
                existing_content += f"\n\n===== {date_str} {hour_str}:00 =====\n\n"
                existing_content += content
                print(f"{date_str} {hour_str}시: 데이터 재수집 성공")
            else:
                print(f"{date_str} {hour_str}시: 데이터 재수집 실패")

            # API 과부하 방지를 위한 지연
            time.sleep(1)

        # 업데이트된 내용 저장
        with open(existing_file, 'w', encoding='utf-8') as f:
            f.write(existing_content)

        # 재시도 파일 삭제
        os.remove(file_path)


def collect_all_data_for_2024(output_path):
    """
    2024년 1월 1일부터 12월 31일까지의 데이터를 수집하여 txt 파일로 저장
    """
    # 임시 디렉토리 생성
    temp_dir = os.path.join(output_path, "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # 실패한 날짜 목록
    failed_days = []

    # 진행 상황 파일
    progress_file = os.path.join(temp_dir, "progress.json")

    # 이전 진행 상황 확인
    start_date = datetime(2024, 1, 1)
    start_day = 0

    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            if 'last_day' in progress:
                start_day = progress['last_day'] + 1
                print(f"이전 진행 상황에서 계속: {start_day}일차부터 시작")

    # 수집 시작
    print(f"2024년 데이터 수집 시작 (1월 1일 ~ 12월 31일)")
    print(f"저장 경로: {output_path}")
    print("=" * 60)

    # 날짜별 반복 (1월 1일부터 12월 31일까지)
    total_days = 366  # 2024년은 윤년

    for day in range(start_day, total_days):
        # 현재 날짜 계산
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime('%Y%m%d')

        print(f"\n{current_date.strftime('%Y-%m-%d')} ({day + 1}/{total_days}) 데이터 수집 중...")

        # 데이터 수집
        success, _ = collect_day_data(output_path, date_str, temp_dir)

        if not success:
            failed_days.append(date_str)

        # 진행 상황 저장
        with open(progress_file, 'w') as f:
            json.dump({'last_day': day, 'failed_days': failed_days}, f)

        # 10일마다 실패한 날짜 재시도
        if (day + 1) % 10 == 0 and failed_days:
            still_failed = retry_failed_days(output_path, failed_days, temp_dir)
            failed_days = still_failed

            # 진행 상황 업데이트
            with open(progress_file, 'w') as f:
                json.dump({'last_day': day, 'failed_days': failed_days}, f)

    # 모든 날짜 처리 후 실패한 날짜 재시도
    if failed_days:
        print("\n모든 데이터 수집 완료. 실패한 날짜 재시도 중...")
        retry_failed_days(output_path, failed_days, temp_dir)

    # 실패한 시간대 재시도
    retry_failed_hours(output_path, temp_dir)

    # 최종 결과 출력
    print("\n" + "=" * 60)
    print(f"2024년 전체 데이터 수집 완료")
    print(f"저장 경로: {output_path}")


if __name__ == "__main__":
    # 저장 경로 설정
    output_path = r"C:\Users\USER\PycharmProjects\PythonProject1\.venv\weather2"

    print("2024년 6월 8일 AMOS 데이터 수집 시작")
    print("=" * 60)

    # 하루만 수집
    collect_day_data(output_path, "20240608")
