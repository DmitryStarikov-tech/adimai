import os
import re
import sys
import subprocess

def run_code(generated_text):
    if not os.path.exists("data.csv"):
        dummy_data = """date,Temperature (Celsius),Humidity (%),Pressure (hPa),Dewpoint (Celsius)
2026-10-20,10.5,65,1012,4.2
2026-10-21,11.2,60,1010,3.8
2026-10-22,9.8,70,1015,4.5
2026-10-23,12.1,55,1008,3.1
2026-10-24,13.5,50,1005,2.9
2026-10-25,14.0,52,1004,2.5
2026-10-26,12.8,58,1007,3.0
"""
        with open("data.csv", "w", encoding="utf-8") as f:
            f.write(dummy_data)
        print("\n[Песочница]: Создан тестовый файл data.csv")

    code_match = re.search(r"```python(.*?)```", generated_text, re.DOTALL)
    if not code_match:
        print("\n[Система]: Модель не выдала код в блоке ```python.")
        return

    code = code_match.group(1).strip()
    code = code.replace("plt.show()", "plt.savefig('plot.png')")
    
    wrapper = f"""
import pandas as pd
import matplotlib.pyplot as plt
import inspect

{code}

if __name__ == '__main__':
    try:
        df = pd.read_csv('data.csv')
        
        # Ищем функцию, которая начинается с plot_
        functions = [(name, func) for name, func in locals().items() if inspect.isfunction(func) and name.startswith('plot_')]
        
        if functions:
            func_name, func = functions[0]
            print(f"[Песочница]: Вызываю функцию {{func_name}}()...")
            # Передаем путь к файлу, так как модель написала функцию с аргументом file_path
            try:
                func('data.csv')
            except TypeError:
                # Если модель написала функцию без аргументов, пробуем вызвать без них
                func()
        else:
            print("[Песочница]: Функция для графика не найдена, выполняю код как скрипт.")
    except FileNotFoundError:
        print("[Песочница]: ОШИБКА! Файл 'data.csv' не найден.")
    except Exception as e:
        print(f"[Песочница]: ОШИБКА выполнения: {{e}}")
"""
    
    with open("temp_script.py", "w", encoding="utf-8") as f:
        f.write(wrapper)
    
    print("\n[Система]: Запуск кода в песочнице...")
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
