require('dotenv').config();
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const http = require('http');
const { spawn } = require('child_process');
const { ragSearch } = require('./rag');
const { fetchAMOSWeather } = require('./weatherApi');
const SessionManager = require('./session');

const app = express();
const port = process.env.PORT || 3010;

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'localhost';
const OLLAMA_PORT = process.env.OLLAMA_PORT || 11434;
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'mistral';

// 세션 매니저 초기화
const sessionManager = new SessionManager();

app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// 세션 미들웨어 - 클라이언트 요청에 맞게 수정
app.use((req, res, next) => {
  // 다양한 형태의 세션 ID 헤더 확인
  let sessionId = req.headers['x-session-id'] || 
                 req.headers['X-Session-ID'] || 
                 req.headers['X-session-id'] || 
                 req.get('X-Session-ID') ||
                 req.get('x-session-id');
  
  console.log('🔍 세션 ID 확인:', { 
    sessionId, 
    method: req.method,
    url: req.url 
  });
  
  // 세션 ID가 있고 유효한 세션인지 확인
  if (sessionId && sessionManager.getSession(sessionId)) {
    // 기존 세션 사용
    req.sessionId = sessionId;
    res.setHeader('X-Session-ID', sessionId);
    
    // 세션 활동 시간 업데이트
    sessionManager.updateSession(sessionId, {
      lastActivity: new Date()
    });
    
    console.log(`🔄 기존 세션 사용: ${sessionId}`);
  } else {
    // 새 세션 생성 (기존 세션이 없거나 유효하지 않을 때만)
    sessionId = sessionManager.createSession();
    
    // 사용자 정보 저장
    sessionManager.updateSession(sessionId, {
      userInfo: {
        ipAddress: req.ip || req.connection.remoteAddress,
        userAgent: req.get('User-Agent')
      }
    });
    
    req.sessionId = sessionId;
    res.setHeader('X-Session-ID', sessionId);
    console.log(`🆕 새 세션 생성: ${sessionId}`);
  }
  
  next();
});

// SSE 스트림 엔드포인트 - 클라이언트 코드에 맞게 수정
app.get('/stream', (req, res) => {
  const sessionId = req.sessionId;
  
  console.log(`📡 SSE 연결 시작 - 세션: ${sessionId}`);
  
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Expose-Headers': 'X-Session-ID',
    'X-Session-ID': sessionId
  });
  
  // 클라이언트가 기대하는 형식으로 세션 연결 메시지 전송
  res.write(`data: 세션 연결됨: ${sessionId}\n\n`);
  
  // 세션별 클라이언트 등록
  sessionManager.registerClient(sessionId, res);
  
  req.on('close', () => {
    sessionManager.removeClient(sessionId, res);
    console.log(`🔌 클라이언트 연결 해제: ${sessionId}`);
  });
});

// 세션별 SSE 전송 함수
function sendSSE(sessionId, data) {
  sessionManager.sendToSession(sessionId, data);
}

// 날짜 파싱 함수
function parseDate(message) {
  const now = new Date();
  let targetDate = new Date(now);
  let dateFound = false;
  
  console.log('원본 메시지:', message);
  
  // 구체적인 날짜 패턴 매칭 (우선순위 순서로 배치)
  const datePatterns = [
    // YYYYMMDD 형식 (20240101 등)
    /(\d{8})(?=날씨|$)/,
    // YYYY년MM월DD일 (한국어 전체 형식)
    /(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일/,
    // YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    /(\d{4})[-\/.](\d{1,2})[-\/.](\d{1,2})/,
    // MM월DD일 (올해 기준)
    /(\d{1,2})월\s*(\d{1,2})일/,
    // MM-DD, MM/DD, MM.DD (올해 기준)
    /(\d{1,2})[-\/.](\d{1,2})(?![-\/.])/,
    // N일 (이번 달 기준)
    /(\d{1,2})일(?!.*월)/
  ];
  
  for (const pattern of datePatterns) {
    const match = message.match(pattern);
    if (match) {
      console.log('매칭된 패턴:', pattern.source, '결과:', match);
      dateFound = true;
      
      if (pattern.source.includes('\\d{8}')) {
        // YYYYMMDD 형식 (20240101)
        const dateStr = match[1];
        const year = parseInt(dateStr.substring(0, 4));
        const month = parseInt(dateStr.substring(4, 6)) - 1; // 0부터 시작
        const day = parseInt(dateStr.substring(6, 8));
        targetDate = new Date(year, month, day);
      } else if (pattern.source.includes('년.*월.*일')) {
        // YYYY년MM월DD일 형식
        const year = parseInt(match[1]);
        const month = parseInt(match[2]) - 1; // 0부터 시작
        const day = parseInt(match[3]);
        targetDate = new Date(year, month, day);
      } else if (pattern.source.includes('\\d{4}')) {
        // YYYY-MM-DD 형식
        const year = parseInt(match[1]);
        const month = parseInt(match[2]) - 1; // 0부터 시작
        const day = parseInt(match[3]);
        targetDate = new Date(year, month, day);
      } else if (pattern.source.includes('월')) {
        // N월 N일 형식
        const month = parseInt(match[1]) - 1;
        const day = parseInt(match[2]);
        targetDate = new Date(now.getFullYear(), month, day);
      } else if (match.length === 3) {
        // MM-DD 형식
        const month = parseInt(match[1]) - 1;
        const day = parseInt(match[2]);
        targetDate = new Date(now.getFullYear(), month, day);
      } else if (match.length === 2) {
        // N일 형식
        const day = parseInt(match[1]);
        targetDate = new Date(now.getFullYear(), now.getMonth(), day);
      }
      break;
    }
  }
  
  // 구체적인 날짜가 없으면 상대적 날짜 체크
  if (!dateFound) {
    if (/오늘|현재/.test(message)) {
      // 오늘 그대로
    } else if (/내일|다음날/.test(message)) {
      targetDate.setDate(now.getDate() + 1);
    } else if (/모레|다음다음날/.test(message)) {
      targetDate.setDate(now.getDate() + 2);
    } else if (/어제|전날/.test(message)) {
      targetDate.setDate(now.getDate() - 1);
    } else if (/그제|그저께/.test(message)) {
      targetDate.setDate(now.getDate() - 2);
    }
  }
  
  console.log('파싱된 날짜:', targetDate, '날짜 발견:', dateFound);
  return targetDate;
}

// 시간 파싱 함수 (분 단위 추가)
function parseTime(message) {
  const now = new Date();
  // 1분 전 시간 계산
  const oneMinuteBefore = new Date(now.getTime() - 60000); // 60000ms = 1분
  let targetHour = oneMinuteBefore.getHours();
  let targetMinute = oneMinuteBefore.getMinutes();
  
  // 시간 패턴 매칭 (분 단위 포함)
  const timePatterns = [
    // HH:MM 형식 (14:30, 9:15 등)
    /(\d{1,2}):(\d{2})/,
    // 오전/오후 HH:MM
    /(오전|오후)\s*(\d{1,2}):(\d{2})/,
    // 오전/오후 N시 M분
    /(오전|오후)\s*(\d{1,2})시\s*(\d{1,2})분/,
    // N시 M분
    /(\d{1,2})시\s*(\d{1,2})분/,
    // 오전/오후 N시 (분은 00으로 설정)
    /(오전|오후)\s*(\d{1,2})시/,
    // N시 (분은 00으로 설정)
    /(\d{1,2})시(?!간)/
  ];
  
  for (const pattern of timePatterns) {
    const match = message.match(pattern);
    if (match) {
      console.log('시간 매칭:', pattern.source, '결과:', match);
      
      if (pattern.source.includes('오전|오후')) {
        const period = match[1];
        const hour = parseInt(match[2]);
        
        if (match[3]) {
          // 분이 포함된 경우
          targetMinute = parseInt(match[3]);
        } else {
          // 분이 없는 경우 00으로 설정
          targetMinute = 0;
        }
        
        if (period === '오전') {
          targetHour = hour === 12 ? 0 : hour;
        } else {
          targetHour = hour === 12 ? 12 : hour + 12;
        }
      } else if (match.length === 3) {
        // HH:MM 형식 또는 N시 M분
        targetHour = parseInt(match[1]);
        targetMinute = parseInt(match[2]);
      } else if (match.length === 2) {
        // N시 형식 (분은 00)
        targetHour = parseInt(match[1]);
        targetMinute = 0;
      }
      break;
    }
  }
  
  return {
    hour: String(targetHour).padStart(2, '0'),
    minute: String(targetMinute).padStart(2, '0')
  };
}

// 채팅 엔드포인트 - 클라이언트 요청에 맞게 수정
app.post('/chat', async (req, res) => {
  const userMessage = req.body.message;
  const sessionId = req.sessionId;
  
  console.log(`💬 채팅 요청 - 세션: ${sessionId}, 메시지: ${userMessage}`);
  
  // 세션 활동 시간 업데이트
  sessionManager.updateSession(sessionId, {
    lastActivity: new Date()
  });
  
  // 클라이언트가 기대하는 형식으로 응답 (sessionId 포함)
  res.json({ status: 'ok', sessionId: sessionId });

  // 사용자 메시지 세션에 저장
  sessionManager.addMessage(sessionId, {
    type: 'user',
    content: userMessage,
    timestamp: new Date()
  });

  const isWeather = /날씨|기온|비|풍향|습도|기후|기상|바람/.test(userMessage);
  const isGoAround = /고어라운드|go[-\s]?around/i.test(userMessage);

  if (isWeather) {
    console.log(`🌦️ 날씨 요청 처리 - 세션: ${sessionId}`);
    
    // 단순히 "날씨", "현재 날씨" 등의 요청인 경우 현재 시간 사용
    const isSimpleWeatherRequest = /^(날씨|현재\s*날씨|오늘\s*날씨|현재날씨|오늘날씨)$/.test(userMessage.trim());
    
    let targetDate, targetTime;
    
    if (isSimpleWeatherRequest) {
      // 단순 날씨 요청은 1분 전 시간 사용
      const now = new Date();
      const oneMinuteBefore = new Date(now.getTime() - 60000);
      targetDate = oneMinuteBefore;
      targetTime = {
        hour: String(oneMinuteBefore.getHours()).padStart(2, '0'),
        minute: String(oneMinuteBefore.getMinutes()).padStart(2, '0')
      };
    } else {
      // 구체적인 날짜/시간이 포함된 요청
      targetDate = parseDate(userMessage);
      targetTime = parseTime(userMessage);
    }
    
    const year = targetDate.getFullYear();
    const month = String(targetDate.getMonth() + 1).padStart(2, '0');
    const day = String(targetDate.getDate()).padStart(2, '0');
    const dateStr = `${year}${month}${day}`;
    
    const dateDisplay = `${year}년 ${month}월 ${day}일 ${targetTime.hour}시 ${targetTime.minute}분`;
    console.log('날씨 조회 요청:', { 
      sessionId,
      userMessage, 
      dateStr, 
      targetTime, 
      dateDisplay, 
      isSimpleWeatherRequest 
    });
    
    sendSSE(sessionId, `🌦️ ${dateDisplay} 날씨 정보를 조회하고 있습니다...\n`);

    // API 호출을 위한 시간 형식 (YYYYMMDDHHMM)
    const targetDateTime = dateStr + targetTime.hour + targetTime.minute;

    try {
      const weatherData = await fetchAMOSWeather(targetDateTime);
      sendSSE(sessionId, weatherData);
      
      // 날씨 데이터를 세션에 저장
      sessionManager.setWeatherData(sessionId, {
        dateTime: targetDateTime,
        data: weatherData,
        requestTime: new Date()
      });
      
      // 봇 응답 세션에 저장
      sessionManager.addMessage(sessionId, {
        type: 'bot',
        content: weatherData,
        category: 'weather',
        timestamp: new Date()
      });
      
    } catch (err) {
      console.error('날씨 API 오류:', err);
      const errorMsg = `❌ 날씨 정보 조회 중 오류가 발생했습니다. (요청: ${dateStr} ${targetTime.hour}:${targetTime.minute})\n`;
      sendSSE(sessionId, errorMsg);
      
      // 과거 날짜인 경우 안내 메시지 추가
      const now = new Date();
      const todayStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
      if (dateStr < todayStr) {
        const pastDataMsg = '📅 과거 날씨 데이터는 제공되지 않을 수 있습니다.\n';
        sendSSE(sessionId, pastDataMsg);
      }
      
      // 에러 메시지 세션에 저장
      sessionManager.addMessage(sessionId, {
        type: 'bot',
        content: errorMsg,
        category: 'error',
        timestamp: new Date()
      });
    }

    sendSSE(sessionId, '[DONE]');
    return;
  }

  if (isGoAround) {
    console.log(`🛫 고어라운드 요청 처리 - 세션: ${sessionId}`);
    
    sendSSE(sessionId, '🛫 고어라운드 분석을 시작합니다. Python 모델을 실행 중입니다...\n');

    const py = spawn('python3', [
      '/home/kng/Captain/integrated_model.py'
    ]);

    let goAroundResult = '';

    py.stdout.on('data', (data) => {
      const output = data.toString();
      sendSSE(sessionId, output);
      goAroundResult += output;
    });

    py.stderr.on('data', (data) => {
      const error = `❌ Python 에러: ${data}`;
      console.error(error);
      sendSSE(sessionId, error);
      goAroundResult += error;
    });

    py.on('close', (code) => {
      const finalMsg = code === 0 ? 
        '✅ Python 분석이 완료되었습니다.' : 
        `❌ Python 프로세스가 비정상 종료 (code: ${code})`;
      
      sendSSE(sessionId, finalMsg);
      
      // 고어라운드 결과를 세션에 저장
      sessionManager.setGoAroundData(sessionId, {
        result: goAroundResult,
        exitCode: code,
        requestTime: new Date()
      });
      
      // 봇 응답 세션에 저장
      sessionManager.addMessage(sessionId, {
        type: 'bot',
        content: goAroundResult + finalMsg,
        category: 'goaround',
        timestamp: new Date()
      });
      
      sendSSE(sessionId, '[DONE]');
    });

    return;
  }

  // 일반 질문 처리 - 순수 Ollama만 사용
  console.log('📝 일반 질문 처리:', { sessionId, userMessage });
  
  const messages = [
    {
      role: 'system',
      content: `한국어로만 대답해. 영어금지. 짧게 답변해. 질문하지 마. 설명하지 마. 이것에 대해 언급하지마.`
    },
    { role: 'user', content: userMessage }
  ];

  const options = {
    hostname: OLLAMA_HOST,
    port: OLLAMA_PORT,
    path: '/api/chat',
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  };

  const requestBody = JSON.stringify({
    model: OLLAMA_MODEL,
    messages,
    stream: true,
    options: {
      temperature: 0.1,
      seed: 42,
      repeat_penalty: 1.3,
      num_predict: 50,
      stop: ['?', '당신', '지시', '규칙', '예를', '설명', '문법', '구조', '따라서', '그래서', '왜냐하면']
    }
  });

  let botResponse = '';

  const ollamaReq = http.request(options, ollamaRes => {
    ollamaRes.on('data', chunk => {
      const lines = chunk.toString().split('\n').filter(Boolean);
      for (const line of lines) {
        try {
          const parsed = JSON.parse(line);
          const content = parsed?.message?.content;
          if (content) {
            sendSSE(sessionId, content);
            botResponse += content;
          }
        } catch (e) { }
      }
    });

    ollamaRes.on('end', () => {
      // 봇 응답 세션에 저장
      if (botResponse) {
        sessionManager.addMessage(sessionId, {
          type: 'bot',
          content: botResponse,
          category: 'general',
          timestamp: new Date()
        });
      }
      
      sendSSE(sessionId, '[DONE]');
    });

    ollamaRes.on('error', (e) => {
      console.error('Ollama 오류:', e);
      sendSSE(sessionId, '[ERROR]');
    });
  });

  ollamaReq.on('error', (e) => {
    console.error('Ollama 요청 에러:', e.message);
    sendSSE(sessionId, 'LLM 서버에 연결할 수 없습니다.');
    sendSSE(sessionId, '[ERROR]');
  });

  ollamaReq.write(requestBody);
  ollamaReq.end();
});

// 세션 관리 엔드포인트들
app.get('/api/session/info', (req, res) => {
  const session = sessionManager.getSession(req.sessionId);
  if (session) {
    res.json({
      sessionId: session.id,
      createdAt: session.createdAt,
      lastActivity: session.lastActivity,
      messageCount: session.messages.length,
      hasWeatherData: !!session.weatherData,
      hasGoAroundData: !!session.goAroundData
    });
  } else {
    res.status(404).json({ error: 'Session not found' });
  }
});

// 세션 유효성 검사 - 클라이언트 코드에 맞게 수정
app.get('/api/session/check', (req, res) => {
  const sessionId = req.headers['x-session-id'] || 
                   req.headers['X-Session-ID'] || 
                   req.headers['X-session-id'] || 
                   req.get('X-Session-ID') ||
                   req.get('x-session-id');
  
  console.log('🔍 세션 체크:', { sessionId });
  
  if (!sessionId) {
    return res.json({ valid: false, reason: 'No session ID' });
  }
  
  const session = sessionManager.getSession(sessionId);
  if (!session) {
    return res.json({ valid: false, reason: 'Session not found' });
  }
  
  res.json({ 
    valid: true, 
    sessionId: sessionId,
    messageCount: session.messages.length,
    lastActivity: session.lastActivity
  });
});

// 세션 히스토리 엔드포인트
app.get('/api/session/history', (req, res) => {
  const messages = sessionManager.getMessageHistory(req.sessionId);
  res.json(messages);
});

// 세션 삭제 엔드포인트
app.delete('/api/session', (req, res) => {
  const deleted = sessionManager.deleteSession(req.sessionId);
  res.json({ deleted });
});

// 관리자용 통계 엔드포인트
app.get('/api/admin/stats', (req, res) => {
  const stats = sessionManager.getStats();
  res.json(stats);
});

// 서버 종료 시 세션 정리
process.on('SIGINT', () => {
  console.log('🔄 서버 종료 중...');
  sessionManager.destroy();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('🔄 서버 종료 중...');
  sessionManager.destroy();
  process.exit(0);
});

app.listen(port, () => {
  console.log(`✅ 서버 실행 중: http://10.100.54.111:${port}`);
  console.log(`📊 세션 관리 활성화`);
});