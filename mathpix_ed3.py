# -*- coding: utf-8 -*-
import requests
import base64
import json
import cv2
import numpy as np
import re
import subprocess
from gtts import gTTS
from sympy import symbols, sympify, integrate, diff, solve, fourier_transform, exp, pi
from sympy.abc import x, t, w
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

transformations = standard_transformations + (implicit_multiplication_application,)

def safe_parse(expr_str):
    return parse_expr(expr_str, transformations=transformations)

def run_camera_capture():
    print("[카메라 실행 중] (q를 누르면 수식 캡처)")

cap = cv2.VideoCapture(2)
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
    functions = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln', 'exp', 'Heaviside']

    # 1) 함수명 임시 마스킹 (예: @@sin@@)
    for func in functions:
        expr = re.sub(rf'\b{func}\b', f'@@{func}@@', expr)

    # 2) 숫자와 문자 사이 곱셈 추가 (3x → 3*x)
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    
    # 3) 함수명 뒤 괄호 앞에 곱셈 넣지 않도록 처리
    #    함수명 마스킹 덕분에 여기서는 그냥 문자와 '(' 사이에 곱셈 추가 가능
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)

    # 4) 문자 사이 공백 있는 경우 곱셈 추가 (a b → a*b)
    expr = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1*\2', expr)

    # 5) 임시 마스킹 복원
    for func in functions:
        expr = expr.replace(f'@@{func}@@', func)

    # 6) 함수명 바로 뒤 괄호로 되돌리기 (예: sin*(x) → sin(x))
    #    'func*(' → 'func('
    for func in functions:
        expr = re.sub(rf'{func}\*\(', f'{func}(', expr)

    return expr



def convert_trig_functions(expr):
    trig_patterns = {
        r"\\sin": "sin",
        r"\\cos": "cos",
        r"\\tan": "tan",
        r"\\cot": "cot",
        r"\\sec": "sec",
        r"\\csc": "csc"
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
    
def add_parentheses_to_functions(expr):
    functions = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln', 'exp', 'Heaviside']
    for func in functions:
        # 함수명 뒤에 괄호가 없고, 바로 변수나 숫자 오는 경우 괄호 씌움
        pattern = rf"{func}(?!\s*\()(\s*[a-zA-Z0-9_]+)"
        repl = rf"{func}(\1)"
        expr = re.sub(pattern, repl, expr)
    return expr

def convert_frac(expr):
# \frac{a}{b} → (a)/(b)
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
    # LaTeX 함수명 괄호 여부 상관없이 모두 치환
    expr = expr.replace("\\sin", "sin")
    expr = expr.replace("\\cos", "cos")
    expr = expr.replace("\\tan", "tan")
    expr = expr.replace("\\cot", "cot")
    expr = expr.replace("\\sec", "sec")
    expr = expr.replace("\\csc", "csc")
    
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

    # 정규식: \int 이후의 식과 d x 사이를 최대한 넓게 잡음
    match = re.match(r"\\int\s*(.+)\s*d\s*x", latex_expr)
    if not match:
        print("> 부정적분 형태 아님")
        return

    body = match.group(1).strip()

    # 전처리 전 디버깅용 출력
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
    
    
    # u(t) → Heaviside(t)
    body = body.replace("u(t)", "Heaviside(t)")
    body = preprocess_expr(body)

    expr = safe_parse(body)
    result = fourier_transform(expr, t, w, noconds=True).doit()
    print("> 함수:", expr)
    print("> 푸리에 변환:", result)
    return result

def create_prompt_for_gemma(expr_line, cmd_line, result):
    prompt = f"""다음 수학 수식을 {cmd_line}하세요.

수식:
{expr_line}

계산 결과:
{result}

꼭 수식의 풀이가 계산결과와 일치하게 결론을 내세요.
풀이 과정을 네 문장 이하로 자세히 설명해 주세요.
식이 있는 경우 무조건 LaTeX로 변환해서 설명하세요.
'^2'나 '**2'는 '~의 제곱'으로 변경하세요.
"""
    return prompt

def ask_gemma_with_ollama(prompt):
    print("\n{Gemma에 질문 전송 중...}\n")
    try:
        result = subprocess.run(
            ['ollama', 'run', 'gemma3:4b'],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        if result.returncode == 0:
            gemma_text = result.stdout.decode('utf-8')
            print("> Gemma 응답:\n", gemma_text)
            process_and_tts(gemma_text)
        else:
            print("> Gemma 실행 오류:", result.stderr.decode('utf-8'))
    except Exception as e:
        print("> subprocess 오류:", e)

#  SRE + gTTS 연동
def clean_speech_text(text):
    text = text.replace('left parenthesis', '')
    text = text.replace('right parenthesis', '')
    text = ' '.join(text.split())
    return text

def latex_to_speech(latex_str):
    try:
        proc = subprocess.run(
            ['node', 'test_sre.js'],
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
def handle_latex3(latex):
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

    if result is not None:
        prompt = create_prompt_for_gemma(expr_line, cmd_line, result)
        ask_gemma_with_ollama(prompt)


#  전체 실행
if __name__ == "__main__":
    run_camera_capture()
    image_path = "capture.jpg"
    latex = get_latex_from_mathpix(image_path)
    handle_latex3(latex)

