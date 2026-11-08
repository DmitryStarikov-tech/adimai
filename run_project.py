import os
import sys
import tty
import termios
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
from sandbox import run_code

class PLELayer(nn.Module):
    def __init__(self, original_layer, model_ref, vocab_size, d_model, ple_dim):
        super().__init__()
        self.original_layer = original_layer
        self._model_ref = [model_ref] 
        self.vocab_size = vocab_size 
        device = next(original_layer.parameters()).device
        
        self.ple_embed = nn.Embedding(vocab_size, ple_dim, dtype=torch.float16).to(device)
        self.w_gate = nn.Linear(d_model, ple_dim, bias=False, dtype=torch.float16).to(device)
        self.w_proj = nn.Linear(ple_dim, d_model, bias=False, dtype=torch.float16).to(device)

    def forward(self, hidden_states, *args, **kwargs):
        layer_outputs = self.original_layer(hidden_states, *args, **kwargs)
        out_hidden = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            
        seq_len = out_hidden.shape[1]
        input_ids = self._model_ref[0].current_input_ids[:, -seq_len:].to(out_hidden.device)
        input_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)
        
        ple_vec = self.ple_embed(input_ids) 
        gate = torch.sigmoid(self.w_gate(out_hidden)) 
        ple_out = self.w_proj(ple_vec * gate) 
        out_hidden = out_hidden + ple_out
        
        return (out_hidden,) + layer_outputs[1:] if isinstance(layer_outputs, tuple) else out_hidden

class EarlyExitRouter(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(self.norm(hidden_states))

def inject_ple(model, vocab_size=None, ple_dim=96):
    d_model = model.config.hidden_size
    vocab_size = vocab_size or model.config.vocab_size
    original_forward = model.forward
    def forward_with_input_ids(*args, **kwargs):
        if 'input_ids' in kwargs: model.current_input_ids = kwargs['input_ids']
        elif len(args) > 0: model.current_input_ids = args[0]
        return original_forward(*args, **kwargs)
    model.forward = forward_with_input_ids
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = PLELayer(layer, model, vocab_size, d_model, ple_dim)
    return model

def inject_routers(model, router_layers=[8, 16, 24]):
    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    model.early_routers = nn.ModuleDict({
        str(idx): EarlyExitRouter(d_model, vocab_size).half().to(model.device) for idx in router_layers
    })
    model.router_layers = router_layers
    return model

def get_user_choice():
    sys.stdout.write("\n\033[33m" + "─" * 65 + "\033[0m\n")
    sys.stdout.write("Security guide\n\n")
    sys.stdout.write("Quick safety check: Модель сгенерировала код. Хотите запустить его в безопасной песочнице?\n\n")

    options = [
        "1. Да, я доверяю этому коду (Запустить)",
        "2. Нет, вернуться к модели (Отменить)"
    ]
    selected = 0
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            for i, opt in enumerate(options):
                prefix = "❯ " if i == selected else "  "
                sys.stdout.write(f"\r\033[K{prefix}{opt}\n")
            
            sys.stdout.write(f"\033[{len(options)}A")
            sys.stdout.flush()
            
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                next1 = sys.stdin.read(1)
                next2 = sys.stdin.read(1)
                if next1 == '[':
                    if next2 == 'A': selected = (selected - 1) % len(options)
                    elif next2 == 'B': selected = (selected + 1) % len(options)
            elif ch in ('\n', '\r'):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
    sys.stdout.write("\r\033[6A\033[J")
    sys.stdout.flush()
    
    return selected == 0

def main():
    repo_id = "StarikovDima/adimai"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    model = AutoModelForCausalLM.from_pretrained(
        repo_id, 
        quantization_config=quantization_config,
        device_map="auto"
    )

    model = inject_ple(model)
    ple_path = hf_hub_download(repo_id=repo_id, filename="ple_weights.pt")
    model.load_state_dict(torch.load(ple_path, map_location=model.device), strict=False)

    model = inject_routers(model)
    routers_path = hf_hub_download(repo_id=repo_id, filename="routers_weights.pt")
    routers_state = torch.load(routers_path, map_location=model.device)
    routers_state = {k: v.half() for k, v in routers_state.items()}
    model.early_routers.load_state_dict(routers_state)

    system_prompts = {
        "default": "Ты полезный ИИ-ассистент AdimAI. Отвечай кратко и по делу.",
        "analyst": "Ты дата-саентист. Пиши сплошной Python-код без функций (без def). Если строишь график, всегда завершай код командой plt.show().",
        "planner": "Ты системный аналитик и планировщик проектов. Твоя задача — разбивать сложные запросы пользователя на четкие, логичные, пошаговые планы действий. Не пиши код. Выдавай структурированные списки задач."
    }
    current_skill = "default"

    print("\nГотово! Терминал запущен.")
    print("Для выхода напиши 'exit' ")

    while True:
        try:
            user_input = input(f"\033[36m[{current_skill}]\033[0m Вы: ")
        except EOFError:
            break

        if user_input.lower() in ['exit', 'quit', '/exit']:
            print("Завершение работы...")
            break
            
        if user_input.startswith('/skill '):
            skill_name = user_input.split(' ', 1)[1].strip()
            if skill_name in system_prompts:
                current_skill = skill_name
                print(f"\n\033[32m[Система]: Скилл '{skill_name}' успешно активирован!\033[0m\n")
            else:
                print(f"\n\033[31m[Система]: Скилл '{skill_name}' не найден.\033[0m\n")
            continue

        if not user_input.strip():
            continue
            
        full_prompt = f"<|im_start|>system\n{system_prompts[current_skill]}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=300, 
            temperature=0.7, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"\nAdimai: {response}")

        if "```python" in response:
            should_run = get_user_choice()
            
            if should_run:
                run_code(response)
            else:
                print(" [Система]: Запуск отменен.")
            print("\n")

if __name__ == "__main__":
    main()
