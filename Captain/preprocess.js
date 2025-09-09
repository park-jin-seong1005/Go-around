const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const iconv = require('iconv-lite'); // npm install iconv-lite

function csvToJson(csvFilePath, jsonFilePath) {
  const results = [];

  fs.createReadStream(csvFilePath)
    .pipe(iconv.decodeStream('euc-kr')) // 여기서 EUC-KR 디코딩 후 UTF-8 스트림으로 변환
    .pipe(csv())
    .on('data', (data) => results.push(data))
    .on('end', () => {
      fs.writeFileSync(jsonFilePath, JSON.stringify(results, null, 2));
      console.log(`CSV를 JSON으로 변환 완료: ${jsonFilePath}`);
    });
}

// 실행 예시
const csvPath = path.join(__dirname, 'data', 'weather.csv');
const jsonPath = path.join(__dirname, 'data', 'weather.json');

csvToJson(csvPath, jsonPath);
