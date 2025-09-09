const fs = require('fs');
const path = require('path');
const axios = require('axios');
require('dotenv').config();

async function fetchAMOSWeather(tm , dtm=1 , stn =182 ) {
  const url = 'https://apihub.kma.go.kr/api/typ01/url/amos.php?tm=202211301200&dtm=60&help=1&authKey=vbLwQ6jMTmuy8EOozJ5rew';

  try {
    const res = await axios.get(url, {
      params: {
        authKey: process.env.KMA_API_KEY,
        tm,
        dtm,
        stn,
        help: '1',
      },
      responseType: 'text',
    });

    const raw = res.data;
    const lines = raw.split('\n').map(line => line.trim()).filter(line => line);

    // 특정 tm 포함된 라인 찾기
    const matchedLine = lines.find(line => line.includes(tm));
    if (!matchedLine) {
      console.warn(`❗ '${tm}' 과 일치하는 줄을 찾을 수 없습니다.`);
      return `📭 관측시간 ${tm}의 날씨 데이터를 찾을 수 없습니다.`;
    }

    const parts = matchedLine.split(/\s+/);
    if (parts.length < 27) {
      return '⚠️ 데이터 파싱에 실패했습니다.';
    }

    const weather = {
      time: parts[1],
      temperature: (parseInt(parts[7], 10) / 10).toFixed(1),
      humidity: parts[9],
      windDirection: parts[21],
      windSpeed: (parseInt(parts[24], 10) / 10).toFixed(1),
      rainfall: (parseInt(parts[12], 10) / 10).toFixed(1),
    };

    const weather2 = {
      CH_MIN: (parseInt(parts[6], 10)).toFixed(1),
      TA: (parseInt(parts[7], 10)).toFixed(1),
      TD: (parseInt(parts[8], 10)).toFixed(1),
      PS: (parseInt(parts[10], 10)).toFixed(1),
      PA: (parseInt(parts[11], 10)).toFixed(1),
      RN: (parseInt(parts[12], 10)).toFixed(1),
      WS02: (parseInt(parts[19], 10)).toFixed(1),
      WS02_max: (parseInt(parts[20], 10)).toFixed(1),
      WD10: (parseInt(parts[22], 10)).toFixed(1),
      WD10_max: (parseInt(parts[23], 10)).toFixed(1),
      WS10: (parseInt(parts[24], 10)).toFixed(1),
      WS10_max: (parseInt(parts[25], 10)).toFixed(1)
    };

    // CSV 저장
    const csvHeader = '최저운고(m),기온(0.1c),이슬점온도(0.1c),해면기압 QFF (0.1hPa),현지기압 QFE (0.1hPa),강수량 (0.1mm),2분 평균풍속 (0.1m/s),2분 최대풍속 (0.1m/s),10분 평균풍향 (degree),10분 최우풍향: 평균풍향을 기준으로 시계방향으로 스캔하여 첫번째 값 (degree),10분 평균풍속 (0.1m/s),10분 최대풍속 (0.1m/s)';
    const csvLine = `${weather2.CH_MIN},${weather2.TA},${weather2.TD},${weather2.PS},${weather2.PA},${weather2.RN},${weather2.WS02},${weather2.WS02_max},${weather2.WD10},${weather2.WD10_max},${weather2.WS10},${weather2.WS10_max}`;

    const outputDir = path.join(__dirname, 'output');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    const fileName = `output.csv`;
    const filePath = path.join(outputDir, fileName);

    // 덮어쓰기: 기존 파일 존재 여부 상관없이 새로 작성
    fs.writeFileSync(filePath, csvHeader + '\n' + csvLine + '\n', 'utf-8');

    console.log(`✅ CSV에 데이터 덮어쓰기 완료: ${filePath}`);

    // 보기 좋은 메시지 반환
    return (
      `🌤️ 제주공항 관측 날씨 (${tm})` +
      `🌡️ 기온: ${weather.temperature}°C` +
      `💧 습도: ${weather.humidity}%` +
      `🧭 풍향: ${weather.windDirection}°` +
      `💨 풍속: ${weather.windSpeed}m/s` +
      `🌧️ 강수량: ${weather.rainfall}mm` 
    );
  } catch (err) {
    if (err.response && err.response.status === 401) {
      return '🔐 API 키가 유효하지 않습니다.';
    }
    console.error('AMOS API 호출 오류:', err);
    return '🚨 날씨 정보를 가져오는 데 실패했습니다.';
  }
}

module.exports = { fetchAMOSWeather };
