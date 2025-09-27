import socket
import subprocess

HOST = '0.0.0.0'  # 서버 IP (모든 인터페이스에서 수신)
PORT = 12345      # 포트 번호

def ask_gemma_with_ollama(prompt):
    try:
        result = subprocess.run(
            ['ollama', 'run', 'gemma3:12b'],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=500
        )
        if result.returncode == 0:
            gemma_text = result.stdout.decode('utf-8')
            return gemma_text
        else:
            return f"Gemma 실행 오류: {result.stderr.decode('utf-8')}"
    except Exception as e:
        return f"subprocess 오류: {str(e)}"

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"서버 시작: {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print('클라이언트 연결:', addr)
                try:
                    data = conn.recv(4096).decode('utf-8')
                    if not data:
                        print("빈 데이터 수신, 연결 종료")
                        break

                    print("받은 질문:", data.strip())
                    response = ask_gemma_with_ollama(data.strip())

                    # 응답 전송 (utf-8 인코딩 필수)
                    conn.sendall(response.encode('utf-8'))

                except Exception as e:
                    error_msg = "오류 발생\n"
                    conn.sendall(error_msg.encode('utf-8'))
                    print("오류:", e)

if __name__ == "__main__":
    main()

