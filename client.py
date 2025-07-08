# -*- coding: utf-8 -*-
import socket
import subprocess
import base64
import json
import cv2
import requests
import re
import os
from gtts import gTTS
from latex2mathml.converter import convert as latex2mathml
from sympy import symbols, sympify, integrate, diff, solve, fourier_transform, exp, pi
from sympy.abc import x, t, w
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

HOST = '10.10.15.165'
PORT = 12345

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

def get_latex_from_mathpix(image_path): 
    headers = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "Content-type": "application/json"
    }
    data = {
        "src": "data:image/jpg;base64," + image_to_base64(image_path),
        "formats": ["latex_styled"],
        "ocr": ["math"]
    }
    response = requests.post("https://api.mathpix.com/v3/text", 
    headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json().get("latex_styled", "")
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


def create_prompt_for_gemma(expr_line, cmd_line, result):
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
    pattern = re.compile(r'\$(.+?)\$', re.DOTALL)
    def replacer(m):
        latex = m.group(1).replace('\n', ' ').strip()
        return latex_to_speech(latex)
    return pattern.sub(replacer, text)

import os

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

    # 자동 재생 (Ubuntu/Linux)
    try:
        print(" 음성 자동 재생 중...")
        os.system(f"mpg123 {mp3_path}")
    except Exception as e: 
        print(f" 음성 재생 오류: {e}")

#  최종 수식 파서
def handle_latex(latex):
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
            # 음성까지 처리
            process_and_tts(gemma_response)
        else:
            print("> Gemma 서버 응답 없음.")



#  전체 실행
if __name__ == "__main__":
    run_camera_capture()
    image_path = "capture.jpg"
    latex = get_latex_from_mathpix(image_path)
    handle_latex(latex)
