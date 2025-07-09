# -*- coding: utf-8 -*-
import socket
import subprocess
import base64
import json
import cv2
import requests
import re
import os
import sounddevice as sd
import numpy as np
import wave
import whisper
from scipy.signal import resample_poly
from gtts import gTTS
from latex2mathml.converter import convert as latex2mathml
from sympy import symbols, sympify, integrate, diff, solve, fourier_transform, exp, pi
from sympy.abc import x, t, w
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
# ===== 설정 =====
RATE_IN = 48000           # 실제 마이크 샘플레이트 (보통 48kHz)
RATE_OUT = 16000          # Whisper에 맞는 샘플레이트
CHANNELS = 1
OUTPUT_WAV = "recorded.wav"
OUTPUT_TXT = "transcript.txt"
MODEL_NAME = "small"
#MODEL_NAME = "medium"
GAIN = 3.0                # 녹음 볼륨 증폭 배율

recording = False
audio_data = []
stream = None

HOST = '10.10.15.165'
PORT = 12345

q_flag=1

transformations = standard_transformations + (implicit_multiplication_application,)

def safe_parse(expr_str):
    return parse_expr(expr_str, transformations=transformations)

def run_camera_capture():
    print("[카메라 실행 중] (q를 누르면 수식 캡처)")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = 20.0

    while(cap.isOpened()):
	    ret, frame = cap.read()
	    if ret is False:
		    print("Can't receive frame (stream end?). Exiting ...")
		    break
	    cv2.imshow("Camera", frame)

	    key = cv2.waitKey(1)
	    if key & 0xFF== ord('q'):
		    cv2.imwrite("capture.jpg", frame)
		    break
    cap.release()
    cv2.destroyAllWindows()

APP_ID = "ai_tutor_53fc7f_e30dd0"
APP_KEY = "86bf3ac09af8c882ea559da4ff73c32f5574752960f5b173197e842cb13c822b"

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_latex_from_mathpix(image_path,mode = 'default'): 
    headers = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "Content-type": "application/json"
    }
    if mode == 'default':
        data = {
            "src": "data:image/jpg;base64," + image_to_base64(image_path),
            "formats": ["latex_styled"],
            "ocr": ["math"]
        }
        response = requests.post("https://api.mathpix.com/v3/text", headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json().get("latex_styled", "")
    
    elif mode == 'verify':
        data = {
            "src": "data:image/jpg;base64," + image_to_base64(image_path),
            "formats": ["latex_styled", "text"],  # ← 추가
            "ocr": ["math", "text"]               # ← 추가
        }
        response = requests.post("https://api.mathpix.com/v3/text", headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            res = response.json()
            latex = res.get("latex_styled", "")
            text = res.get("text", "")  # ← 일반 텍스트 포함
            return text + "\n\n" + latex  # ← 두 결과 결합하여 반환
            
    else:
        raise Exception("Mathpix API 오류:", response.text)

def clean_exponents(expr):
    expr = re.sub(r'\*\*\{(\d+)\}', r'**\1', expr)
    expr = re.sub(r'\^(\{(\d+)\})', r'**\2', expr)
    return expr

def insert_multiplication(expr):
    functions = ['sin', 'cos', 'tan', 'cot', 'sec', 
    'csc', 'log', 'ln', 'exp', 'Heaviside']
    for func in functions:
        expr = re.sub(rf'\b{func}\b', f'@@{func}@@', expr)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
    expr = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1*\2', expr)
    for func in functions:
        expr = expr.replace(f'@@{func}@@', func)
    for func in functions:
        expr = re.sub(rf'{func}\*\(', f'{func}(', expr)
    return expr



def convert_trig_functions(expr):
    trig_patterns = {
        r"\\sin": "sin", r"\\cos": "cos", r"\\tan": "tan", 
        r"\\cot": "cot", r"\\sec": "sec", r"\\csc": "csc"
    }
    for latex_func, sympy_func in trig_patterns.items():
        # f-string 없이 순수 문자열로 작성
        pattern = latex_func + r"\s*\(\s*([^\)]+)\s*\)"
        replacement = sympy_func + r"(\1)"
        expr = re.sub(pattern, replacement, expr)
    return expr
    
def convert_degrees(expr):
    expr = re.sub(r"(\d+)\s*(\\\^)?\{?\\circ\}?", r"rad(\1)", expr)
    expr = re.sub(r"(\d+)\s*\u00b0", r"rad(\1)", expr)

    return expr
    
# 함수명 뒤에 괄호가 없고, 바로 변수나 숫자 오는 경우 괄호 씌움

def add_parentheses_to_functions(expr):
    functions = ['sin', 'cos', 'tan', 'cot', 'sec', 
    'csc', 'log', 'ln', 'exp', 'Heaviside']
    for func in functions:
        
        pattern = rf"{func}(?!\s*\()(\s*[a-zA-Z0-9_]+)"
        repl = rf"{func}(\1)"
        expr = re.sub(pattern, repl, expr)
    return expr

# \frac{a}{b} → (a)/(b)

def convert_frac(expr):
    
    pattern = r"\\frac\s*\{([^\}]+)\}\s*\{([^\}]+)\}"
    while re.search(pattern, expr):
        expr = re.sub(pattern, r"(\1)/(\2)", expr)
    return expr

def preprocess_expr(expr):
    expr = expr.replace(" ", "")
    expr = expr.replace("\\pi", "pi")
    expr = expr.replace("\\ln", "log")
    expr = expr.replace("\\left", "").replace("\\right", "")
    
    # e^{...} → E**(...)
    expr = re.sub(r"e\^{([^}]+)}", r"E**(\1)", expr)
    expr = convert_frac(expr) 
    expr = clean_exponents(expr.replace("^", "**"))
    expr = convert_trig_functions(expr)
    expr = convert_degrees(expr)
    expr = insert_multiplication(expr)
    expr = add_parentheses_to_functions(expr)
    
    return expr


def solve_direct_eval(latex_expr):
    print("\n\U0001F4D8 [수식 평가 모드] 수식:", latex_expr)
    try:
        body = preprocess_expr(latex_expr)
        expr = safe_parse(body)
        result = round(expr.evalf(), 2)
        print(f"\n> 계산 결과: {expr} = {result}")
    except Exception as e:
        print("> 수식 평가 실패:", e)
        
def solve_indefinite_integral(latex_expr):
    print("\n\U0001F4D8 [부정적분 모드] 수식:", latex_expr)

    match = re.match(r"\\int\s*(.+)\s*d\s*x", latex_expr)
    if not match:
        print("> 부정적분 형태 아님")
        return

    body = match.group(1).strip()
    print(f"> 원본 추출 식: {body}")
    body = preprocess_expr(body)
    print(f"> 전처리 후 식: {body}")

    try:
        f_expr = safe_parse(body)
    except Exception as e:
        print(f"> 파싱 실패: {e}")
        return

    F = integrate(f_expr, x)
    print(f"> 적분할 함수: f(x) = {f_expr}")
    print(f"\n> 부정적분 결과: ∫f(x) dx = {F} + C")
    return F

def solve_definite_integral(latex_expr):
    print("\n[정적분 모드] 수식:", latex_expr)
    latex_expr = latex_expr.replace("\\left", "").replace("\\right", "").replace(" ", "")
    match = re.match(r"\\int_\{(.+?)\}\^\{(.+?)\}(.+?)d\s*x", latex_expr)
    if not match:
        print("> 정적분 형태 아님")
        return

    a = sympify(preprocess_expr(match.group(1)))
    b = sympify(preprocess_expr(match.group(2)))
    body = preprocess_expr(match.group(3))
    f_expr = safe_parse(body)

    print(f"> 정적분 구간: a = {a}, b = {b}")
    print(f"> 적분할 함수: f(x) = {f_expr}")

    # integrate(f, (x, a, b)) 하면 바로 수치값 결과 반환
    result = integrate(f_expr, (x, a, b)).evalf()

    print(f"\n> 정적분 결과: ∫ from {a} to {b} of f(x) dx = {result}")
    return result

def solve_derivative(latex_expr):
    print("\n[미분 모드] 수식:", latex_expr)
    match = re.search(r"f\(x\)=(.+)", latex_expr)
    if not match:
        match = re.search(r"=([^=]+)", latex_expr)
    if not match:
        body = latex_expr
    else:
        body = match.group(1)
    body = preprocess_expr(body)
    expr = safe_parse(body)
    print("> 함수:", expr)
    print("> 도함수:", diff(expr, x))
    return diff(expr, x)
    
def solve_equation(latex_expr):
    print("\n[방정식 모드] 수식:", latex_expr)
    match = re.match(r'(.+)=0', latex_expr.replace(" ", ""))
    if not match:
        print("> 방정식 형태 아님")
        return
    left = insert_multiplication(clean_exponents(match.group(1).replace("^", "**")))
    expr = sympify(left)
    print("> 방정식:", expr, "= 0")
    print("> 해:", solve(expr, x))
    return solve(expr, x)

def solve_fourier(latex_expr):
    print("\n[푸리에 변환 모드] 수식:", latex_expr)
    pattern = r"\\mathcal{F}\\left\{(.+?)\\right\}"
    match = re.search(pattern, latex_expr, flags=re.DOTALL)
    if match:
        body = match.group(1)
    else:
        body = latex_expr

    body = body.replace("u(t)", "Heaviside(t)")
    body = preprocess_expr(body)
    print(f"전처리된 식: {body}")

    try:
        expr = safe_parse(body)
    except Exception as e:
        print(f" 파싱 실패: {e}")
        return

    try:
        result = fourier_transform(expr, t, w, noconds=True).doit()
        print("푸리에 변환:", result)
        return result
    except Exception as e:
        print(f"푸리에 변환 실패: {e}")
        return


def create_prompt_for_gemma(expr_line, cmd_line, result, following, mode = 'default'):
    if mode == 'default':
        prompt = f"""Solve the following math problem by performing {cmd_line}.

Problem:
{expr_line}

Result:
{result}

You MUST follow ALL of these rules. NO exceptions.

1. Write ALL mathematical expressions in LaTeX format, and enclose every math expression, even simple ones like $x=2$, inside dollar signs ($...$).
2. The conclusion of your explanation MUST exactly match the result above.
3. Explain the solution clearly in NO MORE THAN four sentences.
4. Replace any '^2' or '**2' with the Korean phrase “~의 제곱”.
5. Your entire response MUST be written in Korean.
6. Avoid redundant explanations. Keep each sentence short and focused.

Example format:
1. [State the given formula.]
2. [Explain the calculation method simply.]
3. [Show substitution or intermediate calculation.]
4. [Clearly state the final result.]

Follow the rules strictly and respond accordingly.
"""
    elif mode == 'verify':
    prompt = f"""Judge whether the following solution steps and final answer are mathematically correct.
Problem:
{expr_line}

Solution steps:
{following}

Answer:
{result}

[Instructions]
- Evaluate the **solution steps** and **answer** separately for correctness.  
- If something is wrong, explain **only** the incorrect part.  
- Even if you don’t like the style of the method, judge it as correct if the logic and result are accurate.  
- When checking correctness, ignore calculation order or method style.  
- Write **all mathematical expressions in LaTeX, enclosed in dollar signs ($...$)**.  
- Use **Korean only** for all other text.
"""
    return prompt


def ask_gemma_via_server(prompt, host=HOST, port=PORT):
    print("\n{Gemma 서버에 질문 전송 중...}\n")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall((prompt + "\n").encode('utf-8'))

            data = b""
            while True:
                chunk = s.recv(1024)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:  # 서버가 개행으로 응답 끝 알림 시
                    break
        response = data.decode('utf-8').strip()
        print("> Gemma 응답:\n", response)
        return response
    except Exception as e:
        print("> 서버 통신 오류:", e)
        return None

#  SRE + gTTS 연동
def clean_speech_text(text):
    text = text.replace('left parenthesis', '')
    text = text.replace('right parenthesis', '')
    text = ' '.join(text.split())
    return text

def latex_to_speech(latex_str):
    try:
        proc = subprocess.run(
            ['node', 'sre.js'],
            input=latex_str.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        output = proc.stdout.decode()
        if proc.returncode != 0:
            print(f"[SRE 변환 실패]: {proc.stderr.decode()}")
            return latex_str

        match = re.search(r'\[출력 음성 텍스트\]: (.+)', output)
        if match:
            return clean_speech_text(match.group(1).strip())
        else:
            return output.strip()
    except Exception as e:
        print(f"[SRE 호출 오류]: {e}")
        return latex_str

def process_text(text):
	# 파일에 저장
	if q_flag:
		with open("gemma3_answer.txt", "w", encoding="utf-8") as f:
			f.write(text)

	pattern = re.compile(r'\$(.+?)\$', re.DOTALL)
	def replacer(m):
		latex = m.group(1).replace('\n', ' ').strip()
		return latex_to_speech(latex)
	return pattern.sub(replacer, text)

def process_and_tts(raw_text):
    processed_text = process_text(raw_text)
    processed_text = processed_text.replace('같다', '는')
    processed_text = processed_text.replace('닷', '곱하기')
    print("> 설명 자막:", processed_text)

    # 저장
    tts = gTTS(text=processed_text, lang='ko')
    mp3_path = 'output_sre_gtts.mp3'
    tts.save(mp3_path)
    print(f" {mp3_path} 생성 완료!")

    # 텍스트 파일로 저장
    txt_path = 'output_sre_gtts.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    print(f" {txt_path} 생성 완료!")

    # 자동 재생 (Ubuntu/Linux)
    try:
        print(" 음성 자동 재생 중...")
        os.system(f"mpg123 {mp3_path}")
    except Exception as e: 
        print(f" 음성 재생 오류: {e}")
    
    
    ################
    
def create_prompt_for_gemma_to_ask():
    # 명령어와 이전 output 읽기
    with open("transcript.txt", "r", encoding="utf-8") as f:
        command = f.read().strip()
    with open("gemma3_answer.txt", "r", encoding="utf-8") as f:
        previous_output = f.read().strip()

    # 명령어에 따른 프롬프트 작성
    command = command.replace(" ", "").lower()
    if command == "정답알려줘" or command =="정답알려줘.":
        instruction = "다음 문제에 대한 정답만 간결하게 LaTeX 수식으로 작성해줘."
    elif command == "설명해줘" or command == "설명해줘.":
        instruction = "다음 문제를 단계별로 3~4문장으로 간단히 설명해줘. 수식은 모두 LaTeX로 표현하고 한 문장마다 개행해."
    elif command == "이론알려줘" or command == "이론알려줘.":
        instruction = "다음 문제와 관련된 수학적 이론을 간단히 소개해줘. LaTeX 수식과 함께 최대 5문장 이내로 작성해."
    else:
        print(f"> 알 수 없는 명령어: {command}")
        return None

    prompt = f"""Please respond according to the following instruction:

Instruction:
{instruction}

Problem:
{previous_output}

You MUST follow ALL of these rules. NO exceptions.

1. Write ALL mathematical expressions in LaTeX format and enclose them in dollar signs ($...$).
2. Keep the explanation short and focused, avoid unnecessary words.
3. Entire response MUST be written in Korean.
4. Avoid repeating the question in the answer.

Respond now.
"""
    return prompt

        

def start_recording():
    global recording, audio_data, stream
    print(" 녹음 시작! (다시 s + 엔터 누르면 중지)")
    audio_data = []
    recording = True

    def callback(indata, frames, time, status):
        if recording:
            audio = indata[:, 0].astype(np.float32)  # 1채널 기준
            audio = audio * GAIN
            audio = np.clip(audio, -32768, 32767).astype(np.int16)
            # 48kHz → 16kHz 리샘플링
            audio_16k = resample_poly(audio, up=1, down=3).astype(np.int16)
            audio_data.append(audio_16k)

    stream = sd.InputStream(samplerate=RATE_IN, channels=CHANNELS, dtype='int16', device=2, callback=callback)
    #stream = sd.InputStream(samplerate=RATE_IN, channels=CHANNELS, dtype='int16', callback=callback)
    stream.start()

def stop_recording():
    global recording, stream
    recording = False
    stream.stop()
    stream.close()
    print(" 녹음 중지!")

    audio_np = np.concatenate(audio_data)
    with wave.open(OUTPUT_WAV, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE_OUT)
        wf.writeframes(audio_np.tobytes())
    print(f" 저장됨: {OUTPUT_WAV}")

    transcribe_with_whisper(OUTPUT_WAV, OUTPUT_TXT)

def transcribe_with_whisper(wav_path, txt_path):
    print(f" Whisper {MODEL_NAME} 로드 중…")
    model = whisper.load_model(MODEL_NAME)
    print(f" 음성 인식 중…")
    result = model.transcribe(wav_path, language="ko")

    text = result["text"]
    print(" 인식 결과:")
    print(text)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(f" 텍스트 저장됨: {txt_path}")
    
def ask_ollama():
    prompt = create_prompt_for_gemma_to_ask()
    if not prompt:  # None 또는 빈 문자열 차단
        print("> 명령어에 맞는 프롬프트 생성 실패 또는 빈 프롬프트입니다.")
        return
    gemma_response = ask_gemma_via_server(prompt)
    if gemma_response:
        process_and_tts(gemma_response)
    else:
        print("> Gemma 서버 응답 없음.")

################

#  최종 수식 파서
def handle_latex(latex,mode = 'default'):
    if mode == 'default'
        # 여러 줄 처리
        latex = re.sub(r"\\begin{array}{.*?}", "", latex)
        latex = re.sub(r"\\begin{aligned}", "", latex)
        latex = latex.replace("\\end{array}", "")
        latex = latex.replace("\\end{aligned}", "")
        lines = [line.strip() for line in latex.split("\\\\") if line.strip()]

        if len(lines) < 2:
            print("> 최소 2줄(수식 + 명령어)이 필요합니다.")
            return

        # & 제거, 공백 제거
        expr_line = re.sub(r"&", "", lines[0]).strip()
        cmd_line = re.sub(r"&", "", lines[1]).strip().replace(" ", "").lower()

        print(f"> 수식: {expr_line}")
        print(f"> 명령어: {cmd_line}")
        result = None

        if "diff" in cmd_line:
            result = solve_derivative(expr_line)
        elif "integral" in cmd_line:
            if "\\int_{" in expr_line:
                result = solve_definite_integral(expr_line)
            else:
                result = solve_indefinite_integral(expr_line)
        elif "solve" in cmd_line or "=0" in expr_line:
            if any(func in expr_line for func in ["\\sin", "\\cos", "\\tan", "sin", "cos", "tan"]):
                result = solve_direct_eval(expr_line)
            else:
                result = solve_equation(expr_line)
        elif "fourier" in cmd_line:
            result = solve_fourier(expr_line)
        else:
            print("> 지원하지 않는 명령어:", cmd_line)
            return

        if result is not None:
            # 프롬프트 생성
            prompt = create_prompt_for_gemma(expr_line, cmd_line, result)

            # 서버로 전송
            gemma_response = ask_gemma_via_server(prompt)

            if gemma_response:
                process_and_tts(gemma_response)
            else:
                print("> Gemma 서버 응답 없음.")
                
    elif mode == 'verify':
        latex = re.sub(r"\\begin{array}{.*?}", "", latex)
        latex = re.sub(r"\\begin{aligned}", "", latex)
        latex = latex.replace(r"\end{array}", "")
        latex = latex.replace(r"\end{aligned}", "")
        latex = latex.replace(" ","")
        latex = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', latex)
        # 줄바꿈 명령을 실제 줄바꿈으로 변경
        latex = latex.replace('\\\\', '\n')
        # \[ 와 \] 를 줄바꿈 포함 형태로 분리
        latex = latex.replace(r'\[', '\n')
        latex = latex.replace(r'\]', '\n')
        # LaTeX 명령어끼리 붙어 있을 경우 구분 (ex: \mathrm{C}e → \mathrm{C}\ne)
        latex = re.sub(r'(?<=\})\\(?=\d|[a-zA-Z])', r'\n\\', latex)
        # 여분의 공백 제거
        latex = re.sub(r'[ \t]+', ' ', latex)
        latex = re.sub(r'\n{2,}', '\n', latex)

     	# 사용자가 수식과 결과를 모두 이미지에 쓴 것으로 가정
        #raw_text = latex.replace("\\\\", "\n").replace("&", "")
        latex = latex.replace("\n\n","\n")
        
        # 모든 (와 ) 사이를 \(...\) 으로 묶기
        latex = re.sub(r'\\\(', r'', latex)
        latex = re.sub(r'\\\)', r'', latex)
        latex = latex.strip()
        
        print("\nlatex\n",latex)
        lines = [line.strip() for line in latex.split("\n") if line.strip()]
        # 각 줄을 \( ... \) 로 감싸기
        #lines = [f"\\({line}\\)" for line in lines]
        question = lines[0]
        #question = re.sub(r"&","",lines[0]).strip()

        answer = lines[-1]
        #cmd_line = re.sub(r"&","",lines[1]).strip().replace(" ","")
        
        #following_lines = [line for line in lines[2:] if line.strip() and line.strip() != cmd_line]
        def normalize_key(line):	
            return re.sub(r'[^a-zA-Z0-9]', '', line).lower()
 
        norm_question = normalize_key(question)
        norm_answer = normalize_key(answer)

        seen = set()
        following_lines = []
        for line in lines[1:-1]:
            norm_line = normalize_key(line)
            if norm_line in (norm_question, norm_answer):
                continue
            if norm_line not in seen:
                seen.add(norm_line)
                following_lines.append(line)
            
        question = f"\\({question}\\)"
        following_lines = "\n".join([f"${line}$" for line in following_lines if line])
        answer = f"\\({answer}\\)"
        
        print("\n Question:", question)
        #print("\n cmd_line:", cmd_line)
        print("\n Following Steps:\n", following_lines)
        print("\n answer:", answer)
        
        # 프롬프트 생성 및 Gemma 호출
        prompt = create_prompt_for_gemma(question,"", answer, following_lines, mode)
        ask_gemma_with_ollama(prompt)



#  전체 실행
if __name__ == "__main__":
    run_camera_capture()
    image_path = "capture.jpg"
    print("'q' 또는 'a'를 입력해주세요.")
    mode = 'default'
    q_flag=0
    while True:
        cmd = input("> ").strip()
        if cmd == "q":
            mode = 'default'
            latex = get_latex_from_mathpix(image_path,mode)
            handle_latex(latex, mode)
            print(" 's' + 엔터 → 녹음 시작/중지, 'o' + 엔터 → 확인 'e' + 엔터 → 종료")
        elif cmd == "a":
            mode = 'verify'
            latex = get_latex_from_mathpix(image_path,mode)
            handle_latex(latex, mode)
            print(" 's' + 엔터 → 녹음 시작/중지, 'o' + 엔터 → 확인 'e' + 엔터 → 종료")
        elif cmd == 'e':
            break
        elif cmd == "s":
            if not recording:
                start_recording()
            else:
                stop_recording()
        elif cmd == "o":
            if recording:
                stop_recording()
            print("gemma3에게 질문합니다.")
            ask_ollama()
        elif cmd == "e":
            if recording:
                stop_recording()
            print("종료합니다.")
            break

