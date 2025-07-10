# Team6_mathtutor

팀원: 정민교, 조성현, 윤종민, 박승헌

프로젝트 기간: 25.07.01 ~ 25.07.11 

# 수식 인식+음성 설명+Gemma AI 연동 시스템 설치 가이드

이 프로젝트는 다음 기능을 포함합니다:
- 수식 이미지 캡처 및 Mathpix OCR 인식
- SymPy를 통한 수식 계산 (정적분, 미분, 방정식, 푸리에 변환)
- Speech Rule Engine을 이용한 수식 음성 해석
- gTTS 기반 음성 출력
- Ollama 기반 LLM (Gemma 3:12b) 풀이 생성

# Python 가상환경 생성 및 패키지 설치
### ollama 설치
https://ollama.com/download

### 가상환경 생성
```bash
python3 -m venv .env
source .env/bin/activate
```

### 패키지 설치(라즈베리파이5)
```bash
pip install requests
pip install gtts
pip install sympy
pip install opencv-python==4.5.5.64
pip install numpy==1.23.latex2mathml

# Node.js + SRE
sudo apt install nodejs npm
npm install mathjax-node
npm install speech-rule-engine

# 음성 출력용
sudo apt install mpg123

# OpenCV GUI 에러시
sudo apt install libgtk2.0-dev pkg-config

#Whisper
pip install sounddevice
pip install -U openai-whisper
pip install scipy
```

# 라즈베리파이5 UTF-8설정
```bash
#raspi-config 진입한 다음
sudo raspi-config
# 5. Localisation options -> L1. Locale -> ko_KR.UTF-8 UTF-8추가

#변경 확인
locale

#변경이 안됐을 시 직접 편집
sudo nano /etc/default/locale
```

### 실행 방법
1. server 역할을 할 컴퓨터에서 server.py를 실행
2. client 역할을 할 라즈베리파이에서 client.py를 실행
3. 수식과 요청이 적힌 두 줄 또는 풀이가 적힌 부분을 카메라로 찍은 후(q) 답을 구할지(q) 풀이가 맞는 지 확인하기(a)를 고르기
4. 통신을 통해 Gemma3의 답변을 TTS로 들은 후 나가기(e), 추가 질문을 위한 녹음(s) 선택.
5. 추가 질문을 위한 녹음(s)시 STT된 결과가 맞다면 o로 확정
6. 통신을 통해 Gemma3의 답변을 TTS로 듣기

