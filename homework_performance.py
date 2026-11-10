import torch
import time

print("=== ЗАДАНИЕ 3: Сравнение производительности CPU vs CUDA ===\n")

# Определяем доступность устройств
device_cpu = torch.device('cpu')
device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
has_cuda = (device_cuda.type == 'cuda')

if not has_cuda:
    print("⚠️ Внимание: CUDA (GPU) не обнаружена. Тесты будут запущены только на CPU.\n")

# ==========================================
# 3.1 Подготовка данных (5 баллов)
# ==========================================
sizes = [
    ("64 x 1024 x 1024", (64, 1024, 1024)),
    ("128 x 512 x 512", (128, 512, 512)),
    ("256 x 256 x 256", (256, 256, 256))
]

# ==========================================
# 3.2 Функция измерения времени (5 баллов)
# ==========================================
def measure_time_cpu(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    return (time.time() - start) * 1000  # в миллисекундах

def measure_time_cuda(func, *args, **kwargs):
    if not has_cuda:
        return 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event)  # в миллисекундах

# ==========================================
# 3.3 Сравнение операций (10 баллов)
# ==========================================
operations = [
    ("Матричное умножение", lambda a, b: torch.matmul(a, b)),
    ("Поэлементное сложение", lambda a, b: a + b),
    ("Поэлементное умножение", lambda a, b: a * b),
    ("Транспонирование", lambda a, _: a.T),
    ("Сумма всех элементов", lambda a, _: torch.sum(a))
]

print(f"{'Операция':<22} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
print("-" * 60)

for name, shape in sizes:
    # Генерируем данные на CPU
    a_cpu = torch.rand(shape)
    b_cpu = torch.rand(shape)
    
    # Если есть CUDA, копируем туда
    if has_cuda:
        a_gpu = a_cpu.to(device_cuda)
        b_gpu = b_cpu.to(device_cuda)
    
    for op_name, op_func in operations:
        # Измеряем на CPU
        time_cpu = measure_time_cpu(op_func, a_cpu, b_cpu)
        
        # Измеряем на GPU
        if has_cuda:
            time_gpu = measure_time_cuda(op_func, a_gpu, b_gpu)
            speedup = time_cpu / time_gpu if time_gpu > 0 else 0.0
            gpu_str = f"{time_gpu:<10.2f}"
            speedup_str = f"{speedup:<10.1f}x"
        else:
            gpu_str = "N/A"
            speedup_str = "N/A"
            
        print(f"{op_name:<22} | {time_cpu:<10.2f} | {gpu_str:<10} | {speedup_str}")
    print("-" * 60)

# ==========================================
# 3.4 Анализ результатов (5 баллов)
# ==========================================
print("\n=== 3.4 Анализ результатов ===")
print("1. Наибольшее ускорение на GPU получают операции с высокой вычислительной сложностью, "
      "такие как матричное умножение (matmul), так как GPU имеет тысячи ядер для параллельных вычислений.")
print("2. Простые операции (например, транспонирование или сумма) могут показывать меньшее ускорение "
      "или даже быть медленнее из-за накладных расходов на синхронизацию и архитектуру GPU.")
print("3. С ростом размера матриц (например, 256x256x256) ускорение на GPU обычно возрастает, "
      "так как вычислительная нагрузка перевешивает накладные расходы на запуск ядра.")
print("4. Передача данных между CPU и GPU (метод .to('cuda')) занимает время. "
      "В реальных задачах данные стараются держать на GPU на протяжении всего цикла обучения, "
      "чтобы минимизировать эти издержки.")

print("\n Задание 3 (homework_performance.py) успешно выполнено!")