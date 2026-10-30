import os
import re
import sys
import subprocess

def run_code(response_text):
    match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        match = re.search(r"```\n(.*?)\n```", response_text, re.DOTALL)
        if not match:
            print(" \033[31m[Sandbox]: Не удалось перехватить код в ответе модели.\033[0m")
            return

    code = match.group(1)
    filename = "sandbox_temp.py"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        
        print(f"\033[36m[Sandbox]\033[0m: Файл {filename} создан. Запуск...\n")
        print("\033[2m" + "─" * 65 + "\033[0m")

        subprocess.run([sys.executable, filename], timeout=60)
        
        print("\033[2m" + "─" * 65 + "\033[0m")
        print("\033[32m[Sandbox]: Выполнение успешно завершено.\033[0m")
        
    except subprocess.TimeoutExpired:
        print("\n\033[31m[Sandbox]: Превышен лимит времени в 60 секунд.\033[0m")
    except Exception as e:
        print(f"\n\033[31m[Sandbox]: Системная ошибка при запуске: {e}\033[0m")
    finally:
        if os.path.exists(filename):
            os.remove(filename)
