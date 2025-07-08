# Team6_mathtutor

팀원: 정민교, 조성현, 윤종민, 박승헌

프로젝트 기간: 25.07.01 ~ 25.07.11 

# 수식 인식 + 음성 설명 + Gemma AI 연동 시스템 설치 가이드

이 프로젝트는 다음 기능을 포함합니다:
- 수식 이미지 캡처 및 Mathpix OCR 인식
- SymPy를 통한 수식 계산 (방정식, 삼각함수, 미분, 적분, 푸리에 변환)
- Speech Rule Engine을 이용한 수식 음성 해석
- gTTS 기반 음성 출력
- Ollama 기반 LLM (Gemma 3:4b) 풀이 생성

# Python 가상환경 생성 및 패키지 설치

```bash
python3 -m venv .env
source .env/bin/activate
pip install requests
pip install gtts
pip install sympy
pip install opencv-python==4.5.5.64
pip install numpy==1.23.5

# Node.js + SRE
sudo apt install nodejs npm
npm install mathjax-node
npm install speech-rule-engine

# Gemma 실행용
sudo snap install ollama

# 음성 출력용
sudo apt install mpg123

# OpenCV GUI 에러시
sudo apt install libgtk2.0-dev pkg-config
