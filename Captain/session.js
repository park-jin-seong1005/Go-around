// session.js - 세션 관리 모듈
const { v4: uuidv4 } = require('uuid');

class SessionManager {
  constructor() {
    this.sessions = new Map();
    this.clients = new Map(); // sessionId -> SSE 클라이언트 매핑
    this.cleanupInterval = setInterval(() => this.cleanup(), 30 * 60 * 1000); // 30분마다 정리
  }

  // 새 세션 생성
  createSession() {
    const sessionId = uuidv4();
    const session = {
      id: sessionId,
      createdAt: new Date(),
      lastActivity: new Date(),
      messages: [],
      weatherData: null,
      goAroundData: null,
      userInfo: {
        ipAddress: null,
        userAgent: null
      }
    };
    
    this.sessions.set(sessionId, session);
    console.log(`🆕 새 세션 생성: ${sessionId}`);
    return sessionId;
  }

  // 세션 조회
  getSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.lastActivity = new Date();
      return session;
    }
    return null;
  }
  validateSession(sessionId) {
  if (!sessionId) return false;
  
  const session = this.sessions.get(sessionId);
  if (!session) return false;
  
  // 세션 만료 시간 체크 (예: 24시간)
  const maxAge = 24 * 60 * 60 * 1000;
  const now = new Date();
  
  if (now - session.lastActivity > maxAge) {
    this.deleteSession(sessionId);
    return false;
  }
  
  return true;
}


  // 세션 업데이트
  updateSession(sessionId, data) {
    const session = this.getSession(sessionId);
    if (session) {
      Object.assign(session, data);
      session.lastActivity = new Date();
      return true;
    }
    return false;
  }

  // 메시지 추가
  addMessage(sessionId, message) {
    const session = this.getSession(sessionId);
    if (session) {
      session.messages.push({
        ...message,
        timestamp: new Date()
      });
      session.lastActivity = new Date();
      
      // 메시지 개수 제한 (최대 50개)
      if (session.messages.length > 50) {
        session.messages = session.messages.slice(-50);
      }
      
      return true;
    }
    return false;
  }

  // 세션의 메시지 히스토리 조회
  getMessageHistory(sessionId) {
    const session = this.getSession(sessionId);
    return session ? session.messages : [];
  }

  // 날씨 데이터 저장
  setWeatherData(sessionId, weatherData) {
    const session = this.getSession(sessionId);
    if (session) {
      session.weatherData = weatherData;
      session.lastActivity = new Date();
      return true;
    }
    return false;
  }

  // 고어라운드 데이터 저장
  setGoAroundData(sessionId, goAroundData) {
    const session = this.getSession(sessionId);
    if (session) {
      session.goAroundData = goAroundData;
      session.lastActivity = new Date();
      return true;
    }
    return false;
  }

  // SSE 클라이언트 등록
  registerClient(sessionId, client) {
    if (!this.clients.has(sessionId)) {
      this.clients.set(sessionId, []);
    }
    this.clients.get(sessionId).push(client);
    
    // 클라이언트 종료 시 정리
    client.on('close', () => {
      this.removeClient(sessionId, client);
    });
    
    console.log(`📡 클라이언트 등록: ${sessionId}`);
  }

  // SSE 클라이언트 제거
  removeClient(sessionId, client) {
    const clients = this.clients.get(sessionId);
    if (clients) {
      const index = clients.indexOf(client);
      if (index !== -1) {
        clients.splice(index, 1);
        if (clients.length === 0) {
          this.clients.delete(sessionId);
        }
      }
    }
  }

  // 특정 세션의 클라이언트들에게 SSE 전송
  sendToSession(sessionId, data) {
    const clients = this.clients.get(sessionId);
    if (clients) {
      clients.forEach(client => {
        try {
          client.write(`data: ${data}\n\n`);
        } catch (err) {
          console.error('SSE 전송 오류:', err);
        }
      });
    }
  }

  // 모든 클라이언트에게 전송 (기존 호환성)
  broadcast(data) {
    this.clients.forEach((clients, sessionId) => {
      this.sendToSession(sessionId, data);
    });
  }

  // 세션 삭제
  deleteSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      // 해당 세션의 모든 클라이언트 연결 해제
      const clients = this.clients.get(sessionId);
      if (clients) {
        clients.forEach(client => {
          try {
            client.end();
          } catch (err) {
            console.error('클라이언트 종료 오류:', err);
          }
        });
        this.clients.delete(sessionId);
      }
      
      this.sessions.delete(sessionId);
      console.log(`🗑️ 세션 삭제: ${sessionId}`);
      return true;
    }
    return false;
  }

  // 비활성 세션 정리
  cleanup() {
    const now = new Date();
    const maxAge = 2 * 60 * 60 * 1000; // 2시간
    
    for (const [sessionId, session] of this.sessions) {
      if (now - session.lastActivity > maxAge) {
        this.deleteSession(sessionId);
      }
    }
  }

  // 세션 통계
  getStats() {
    return {
      totalSessions: this.sessions.size,
      activeClients: Array.from(this.clients.values()).reduce((sum, clients) => sum + clients.length, 0),
      sessions: Array.from(this.sessions.values()).map(session => ({
        id: session.id,
        createdAt: session.createdAt,
        lastActivity: session.lastActivity,
        messageCount: session.messages.length,
        hasWeatherData: !!session.weatherData,
        hasGoAroundData: !!session.goAroundData
      }))
    };
  }

  // 종료 시 정리
  destroy() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    
    // 모든 클라이언트 연결 해제
    this.clients.forEach((clients, sessionId) => {
      clients.forEach(client => {
        try {
          client.end();
        } catch (err) {
          console.error('클라이언트 종료 오류:', err);
        }
      });
    });
    
    this.sessions.clear();
    this.clients.clear();
    console.log('🔚 세션 매니저 종료');
  }
}

module.exports = SessionManager;