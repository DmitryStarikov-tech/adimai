import os
import re
import sys
import subprocess

def run_code(generated_text):
    code_match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
    if not code_match:
        print("\n[Система]: Модель не выдала код в блоке ```python.")
        return

    code = code_match.group(1).strip()
    code = code.replace("plt.show()", "plt.savefig('plot.png')")
    
    wrapper = f"""
import pandas as pd
import matplotlib.pyplot as plt

{code}

if __name__ == '__main__':
    try:
        df = pd.read_csv('data.csv')
        import inspect
        current_module = inspect.getmodule(inspect.currentframe())
        functions = inspect.getmembers(current_module, inspect.isfunction)
        called = False
        for name, func in functions:
            if name.startswith('plot_'):
                print(f"[Песочница]: Вызываю функцию {{name}}()...")
                func(df)
                called = True
                break
        if not called:
            print("[Песочница]: Функция для графика не найдена. Код выполнен.")
    except FileNotFoundError:
        print("[Песочница]: ОШИБКА! Файл 'data.csv' не найден.")
    except Exception as e:
        print(f"[Песочница]: ОШИБКА выполнения: {{e}}")
"""
    
    with open("temp_script.py", "w", encoding="utf-8") as f:
        f.write(wrapper)
    
    print("\n[Система]: Запуск кода в песочнице")
    try:
        result = subprocess.run(
            [sys.executable, "temp_script.py"],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0:
            print("Код выполнен успешно!")
            if result.stdout: print(result.stdout)
            if os.path.exists("plot.png"):
                print("График сохранен в файл 'plot.png'!")
                os.system("open plot.png") 
        else:
            print("Ошибка при выполнении:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Ошибка: Код выполнялся слишком долго (больше 15 секунд).")
