import cv2
import pytesseract
import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import re

# SymPy 초기 설정
init_printing()

# 이미지 경로
img_path = 'math2.jpg'  # 분석할 수식 이미지 경로
img = cv2.imread(img_path)

if img is None:
    print("❌ 이미지 로딩 실패")
    exit()


## 전처리

# 1. Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Unsharp Masking (선명도 향상)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

# 3. Canny Edge Detection (경계 검출)
edges = cv2.Canny(sharp, threshold1=50, threshold2=150)

# 4. Morphological close (선 연결 및 외곽 보강)
kernel = np.ones((2, 2), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 5. 이진화 (adaptive threshold) — 텍스트가 잘 보이도록 병합
combined = cv2.bitwise_or(closed, sharp)

# 6. 이미지 확대
scaled = cv2.resize(combined, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# -------------------------------
# ✅ OCR 인식
# -------------------------------
custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789xyzXYZ+-*/=()^.'
raw_text = pytesseract.image_to_string(sharp, config=custom_config)
text = raw_text.strip().replace(' ', '')

print("📝 OCR 인식 결과:", text)

# -------------------------------
# ✅ 수식 전처리 및 보정
# -------------------------------

# ^ → **
text = text.replace('^', '**')

# 곱셈 누락 보정: 2x → 2*x, -6x → -6*x
text = re.sub(r'(?<!\*)\b(\d)([a-zA-Z])', r'\1*\2', text)

# = 인식 오류 대응
text = text.replace(':', '=').replace('＝', '=')

print("🔧 보정된 수식:", text)

# 수식 구조 검사
if not text or '=' not in text or len(text) < 3:
    print("❌ 수식 구조가 올바르지 않음")
    exit()

# -------------------------------
# ✅ SymPy로 수식 파싱 및 풀이
# -------------------------------
try:
    lhs, rhs = text.split('=')

    lhs_expr = parse_expr(lhs)
    rhs_expr = parse_expr(rhs)
    equation = Eq(lhs_expr, rhs_expr)

    print("📐 식:", equation)

    # 변수 추출 및 풀이
    variables = list(equation.free_symbols)
    if not variables:
        print("❌ 변수 없음 - 수치식일 가능성")
        exit()

    solution = solve(equation, variables[0])
    print("✅ 해:", solution)

except Exception as e:
    print("❌ 수식 파싱 실패:", e)

