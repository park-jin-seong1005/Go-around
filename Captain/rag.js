const fs = require('fs');
const path = require('path');

// 데이터 파일 로드
let embeddings = [];
try {
  const dataPath = path.join(__dirname, 'data', 'weather.json');
  if (fs.existsSync(dataPath)) {
    embeddings = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  } else {
    console.log('weather.json 파일이 없습니다. 기본 데이터를 사용합니다.');
    // 기본 날씨 데이터
    embeddings = [
      {
        title: "날씨 기본 정보",
        content: "기온, 습도, 풍속, 풍향, 기압 등의 기상 정보를 제공합니다."
      },
      {
        title: "제주공항 날씨",
        content: "제주공항의 실시간 기상 관측 데이터를 제공합니다."
      }
    ];
  }
} catch (error) {
  console.error('데이터 로드 오류:', error);
  embeddings = [];
}

function ragSearch(question, topK = 3) {
  if (!embeddings || embeddings.length === 0) {
    return '참고할 문서가 없습니다.';
  }

  // 간단한 키워드 매칭 기반 검색
  const keywords = question.toLowerCase()
    .replace(/[^\w\s가-힣]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 1);
  
  const scored = embeddings.map(doc => {
    let score = 0;
    const docText = `${doc.title || ''} ${doc.content || ''}`.toLowerCase();
    
    // 키워드 매칭 점수 계산
    keywords.forEach(keyword => {
      const matches = (docText.match(new RegExp(keyword, 'g')) || []).length;
      score += matches;
    });
    
    // 정확한 질문 포함 시 가산점
    if (docText.includes(question.toLowerCase())) {
      score += 5;
    }
    
    return { ...doc, score };
  });

  // 점수순 정렬
  scored.sort((a, b) => b.score - a.score);
  
  // 상위 K개 선택
  const topDocs = scored.slice(0, topK).filter(d => d.score > 0);
  
  if (topDocs.length === 0) {
    return '관련된 문서를 찾을 수 없습니다.';
  }
  
  return topDocs
    .map(d => `제목: ${d.title || '제목 없음'}\n내용: ${d.content || '내용 없음'}`)
    .join('\n\n---\n\n');
}

module.exports = { ragSearch };