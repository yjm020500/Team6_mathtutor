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
    print("카메라 실행 중... (q를 누르면 수식 캡처, a를 누르면 풀이과정 검산)")

    cap = cv2.VideoCapture(2) #자신의 카메라 번호에 맞게 설정

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

        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame,"q, a, c", (30,80),font, 2, (255,255,255),3)
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)

        if key & 0xFF== ord('q'):
            cv2.imwrite("capture.jpg", frame)
            cap.release()
            cv2.destroyAllWindows()
            return "capture.jpg", "default"
        elif key & 0xFF == ord('a'):
            cv2.imwrite("capture2.jpg",frame)
            cap.release()
            cv2.destroyAllWindows()
            return "capture2.jpg", "verify"
        elif key & 0xff == ord('c'):
            cv2.imwrite("capture3.jpg",frame)
            cap.release()
            cv2.destroyAllWindows()
            return "capture3.jpg", "continue"

APP_ID = "ai_tutor_53fc7f_e30dd0"
APP_KEY = "86bf3ac09af8c882ea559da4ff73c32f5574752960f5b173197e842cb13c822b"

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_latex_from_mathpix(image_path,mode='default'): 
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
        else:
            raise Exception("Mathpix API 오류:", response.text)
    
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
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
    return expr

def solve_integral(latex_expr):
    print("\n[정적분 모드] 수식:", latex_expr)
    latex_expr = latex_expr.replace("\\left", "").replace("\\right", "").replace(" ", "")
    match = re.match(r"\\int_{(\d+)}\^\{(\d+)\}(?:\\left)?\(?(.+?)\)?(?:\\right)?\s*d\s*x", latex_expr)
    if not match:
        print("정적분 형태 아님")
        return
    a = sympify(match.group(1))
    b = sympify(match.group(2))
    body = match.group(3)
    body = insert_multiplication(clean_exponents(body.replace("^", "**")))
    print(f"범위: {a} → {b}")
    print(f"함수: {body}")
    result = integrate(sympify(body), (x, a, b))
    print("결과:", result)
    return result

def solve_derivative2(latex_expr):
    print("\n[미분 모드] 수식:", latex_expr)
    match = re.search(r"f\(x\)=(.+)", latex_expr)
    if not match:
        match = re.search(r"=([^=]+)", latex_expr)
    if not match:
        body = latex_expr
    else:
        body = match.group(1)
    body = clean_exponents(body.replace("^", "**"))
    expr = safe_parse(body)
    print("함수:", expr)
    print("도함수:", diff(expr, x))
    return diff(expr, x)

def solve_equation(latex_expr):
    print("\n[방정식 모드] 수식:", latex_expr)
    match = re.match(r'(.+)=0', latex_expr.replace(" ", ""))
    if not match:
        print("방정식 형태 아님")
        return
    left = insert_multiplication(clean_exponents(match.group(1).replace("^", "**")))
    expr = sympify(left)
    print("방정식:", expr, "= 0")
    print("해:", solve(expr, x))
    return solve(expr, x)

def solve_fourier(latex_expr):
    print("\n[푸리에 변환 모드] 수식:", latex_expr)
    match = re.match(r"\\mathcal{F}\\{(.+)\\}", latex_expr)
    if not match:
        print("푸리에 변환 형태 아님")
        return
    body = insert_multiplication(clean_exponents(match.group(1).replace("^", "**")))
    expr = sympify(body)
    result = fourier_transform(expr, t, w)
    print("함수:", expr)
    print("푸리에 변환:", result)
    return result

def create_prompt_for_gemma(expr_line, cmd_line, result,following, mode = "default"):
    if mode == "default":
       prompt = f"""다음 수학 수식을 {cmd_line}하세요.

 수식:
{expr_line}

계산 결과:
{result}

[지시사항]
- 위 수식을 {cmd_line}하여 계산 결과와 맞는지 확인하라.
- 풀이 과정은 네 문장 이하의 한글로 작성하라.
- 수식이 포함되면 반드시 LaTeX로 표현하라.
- 설명 외에는 아무 말도 하지 마라.
"""

    elif mode == "verify":
        prompt = f"""다음 과정이 수식의 논리와 계산 결과가 정확한지 판단하라:
문제:
{expr_line}
{cmd_line}

풀이 과정:
{following}

[지시사항]
- 풀이과정이 올바른지 판단하되, 잘못된 부분이 있으면 잘못된 부분만 설명해라.
- 계산 방식이 마음에 들지 않아도 정확하면 올바르다고 판단하라.
- 정확성 판단 시 계산 순서, 방법의 스타일 등은 고려하지 마라.
- 수식은 반드시 LaTeX로 작성하라. 그 외에는 한글로만 작성하라.
"""
    #print(prompt)
    return prompt
    
def ask_gemma_with_ollama(prompt):
    print(prompt)
    print("\nGemma에 질문 전송 중...\n")
    try:
        result = subprocess.run(
            ['ollama', 'run', 'gemma3:4b'],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90
        )
        if result.returncode == 0:
            gemma_text = result.stdout.decode('utf-8')
            print("Gemma 응답:\n", gemma_text)
            #process_and_tts(gemma_text)
        else:
            print("Gemma 실행 오류:", result.stderr.decode('utf-8'))
    except Exception as e:
        print("subprocess 오류:", e)
"""
# SRE + gTTS 연동
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
"""

import os

'''
def process_and_tts(raw_text):
    processed_text = process_text(raw_text)
    print("설명 자막:", processed_text)

    # 저장
    tts = gTTS(text=processed_text, lang='ko')
    mp3_path = 'output_sre_gtts.mp3'
    tts.save(mp3_path)
    print(f"{mp3_path} 생성 완료!")

    # 자동 재생 (Ubuntu/Linux)
    try:
        print("음성 자동 재생 중...")
        os.system(f"mpg123 {mp3_path}")
    except Exception as e: 
        print(f"음성 재생 오류: {e}")
'''
# 최종 수식 파서
def handle_latex3(latex,mode="default"):

    if mode == 'default':
    	latex = re.sub(r"\\begin{array}{.*?}", "", latex)
    	latex = re.sub(r"\\begin{aligned}", "", latex)
    	latex = latex.replace("\\end{array}", "")
    	latex = latex.replace("\\end{aligned}", "")
    	lines = [line.strip() for line in latex.split("\\\\") if line.strip()]
    	if len(lines) < 2:
        	print("최소 2줄(수식 + 명령어)이 필요합니다.")
        	return
    

    	expr_line = re.sub(r"&", "", lines[0]).strip()
    	cmd_line = re.sub(r"&", "", lines[1]).strip().replace(" ", "").lower()
    	print(f"수식: {expr_line}")
    	print(f"명령어: {cmd_line}")
    	result = None
    	if "diff" in cmd_line:
    	    result = solve_derivative2(expr_line)
    	elif "integral" in cmd_line:
    	    result = solve_integral(expr_line)
    	elif "solve" in cmd_line or "=0" in expr_line:
    	    result = solve_equation(expr_line)
    	elif "fourier" in cmd_line:
    	    result = solve_fourier(expr_line)
    	else:
    	    print("지원하지 않는 명령어:", cmd_line)

    	if result is not None:
	       prompt = create_prompt_for_gemma(expr_line, cmd_line, result,"",mode)
	       ask_gemma_with_ollama(prompt)
     	    
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

        cmd_line=lines[1]
        #cmd_line = re.sub(r"&","",lines[1]).strip().replace(" ","")
        
        #following_lines = [line for line in lines[2:] if line.strip() and line.strip() != cmd_line]
        def normalize_key(line):	
            return re.sub(r'[^a-zA-Z0-9]', '', line).lower()
 
        norm_question = normalize_key(question)
        norm_cmd_line = normalize_key(cmd_line)

        seen = set()
        following_lines = []
        for line in lines[2:]:
            norm_line = normalize_key(line)
            if norm_line in (norm_question, norm_cmd_line):
                continue
            if norm_line not in seen:
                seen.add(norm_line)
                following_lines.append(line)
            
        question = f"\\({question}\\)"
        cmd_line = f"\\({cmd_line}\\)"
        following_lines = "\n".join([f"${line}$" for line in following_lines if line])
        
        print("\n Question:", question)
        print("\n cmd_line:", cmd_line)
        print("\n Following Steps:\n", following_lines)
        
        # 프롬프트 생성 및 Gemma 호출
        prompt = create_prompt_for_gemma(question,cmd_line,"",following_lines,mode)
        ask_gemma_with_ollama(prompt)
            
    else:
        print("지원하지 않는 모드:", mode)
	

# 전체 실행
if __name__ == "__main__":

    '''
    run_camera_capture()
    image_path = "capture.jpg"
    latex = get_latex_from_mathpix(image_path)
    handle_latex3(latex)
    '''
    

    image_path, mode = run_camera_capture()
    if image_path:
        latex = get_latex_from_mathpix(image_path,mode)
        handle_latex3(latex, mode=mode)
    '''    
    image_path = "capture2.jpg"
    mode = 'verify'
    latex = get_latex_from_mathpix(image_path,mode)
    handle_latex3(latex,mode)
    '''
