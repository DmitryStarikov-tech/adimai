import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download

# --- 1. Классы кастомной архитектуры ---
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
        
        # Проверяем, вернул ли слой кортеж (tuple) или просто тензор
        if isinstance(layer_outputs, tuple):
            out_hidden = layer_outputs[0]
        else:
            out_hidden = layer_outputs
            
        seq_len = out_hidden.shape[1]
        input_ids = self._model_ref[0].current_input_ids[:, -seq_len:].to(out_hidden.device)
        input_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)
        
        ple_vec = self.ple_embed(input_ids) 
        gate = torch.sigmoid(self.w_gate(out_hidden)) 
        ple_out = self.w_proj(ple_vec * gate) 
        
        out_hidden = out_hidden + ple_out
        
        # Возвращаем в том же формате, в котором получили
        if isinstance(layer_outputs, tuple):
            return (out_hidden,) + layer_outputs[1:]
        else:
            return out_hidden

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

# --- 2. Сборка и Запуск ---
def main():
    repo_id = "StarikovDima/adimai"
    print("\n[1/4] Инициализация движка и скачивание базы (4-bit)...")
    
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

    print("[2/4] Загрузка и врезка памяти PLE...")
    model = inject_ple(model)
    ple_path = hf_hub_download(repo_id=repo_id, filename="ple_weights.pt")
    model.load_state_dict(torch.load(ple_path, map_location=model.device), strict=False)

    print("[3/4] Загрузка и врезка роутеров условной глубины...")
    model = inject_routers(model)
    routers_path = hf_hub_download(repo_id=repo_id, filename="routers_weights.pt")
    routers_state = torch.load(routers_path, map_location=model.device)
    # Конвертируем роутеры в float16 для экономии памяти
    routers_state = {k: v.half() for k, v in routers_state.items()}
    model.early_routers.load_state_dict(routers_state)

    print("[4/4] Готово! Терминал запущен.\n")
    print("="*50)
    print("🤖 Qwen-Hybrid (PLE + Условная глубина) активирован.")
    print("Введите 'exit' для выхода.")
    print("="*50)

    while True:
        user_input = input("\nВы: ")
        if user_input.lower() in ['exit', 'quit', '/exit']:
            print("Завершение работы...")
            break
            
        inputs = tokenizer(f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(model.device)
        
        # Стандартная генерация (добавлено do_sample=True)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.7, 
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"\nAdimai: {response}")

if __name__ == "__main__":
    main()
