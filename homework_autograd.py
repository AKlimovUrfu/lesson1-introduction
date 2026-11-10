import torch
import math

print("=== ЗАДАНИЕ 2: Автоматическое дифференцирование ===\n")

# ==========================================
# 2.1 Простые вычисления с градиентами (8 баллов)
# ==========================================
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

# Функция: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2 * x * y * z
f.backward()

print("2.1 Градиенты функции f(x,y,z) = x^2 + y^2 + z^2 + 2xyz")
print(f"df/dx (PyTorch): {x.grad.item():.4f} | Аналитически (2x + 2yz): {2*2 + 2*3*4:.4f}")
print(f"df/dy (PyTorch): {y.grad.item():.4f} | Аналитически (2y + 2xz): {2*3 + 2*2*4:.4f}")
print(f"df/dz (PyTorch): {z.grad.item():.4f} | Аналитически (2z + 2xy): {2*4 + 2*2*3:.4f}\n")


# ==========================================
# 2.2 Градиент функции потерь MSE (9 баллов)
# ==========================================
# Создаем фиктивные данные
x_data = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# Параметры с requires_grad=True
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# Линейная функция и MSE
y_pred = w * x_data + b
mse = ((y_pred - y_true) ** 2).mean()
mse.backward()

print("2.2 Градиенты MSE для y_pred = w*x + b")
print(f"Градиент по w: {w.grad.item():.4f}")
print(f"Градиент по b: {b.grad.item():.4f}\n")


# ==========================================
# 2.3 Цепное правило (8 баллов)
# ==========================================
x_chain = torch.tensor(1.5, requires_grad=True)

# Функция: f(x) = sin(x^2 + 1)
f_chain = torch.sin(x_chain**2 + 1)

# Вычисляем градиент df/dx
f_chain.backward()
grad_pytorch = x_chain.grad.item()

# Проверка с помощью torch.autograd.grad (альтернативный способ)
x_check = torch.tensor(1.5, requires_grad=True)
f_check = torch.sin(x_check**2 + 1)
grad_autograd = torch.autograd.grad(f_check, x_check)[0].item()

# Аналитическая проверка: df/dx = cos(x^2 + 1) * 2x
grad_analytical = math.cos(1.5**2 + 1) * (2 * 1.5)

print("2.3 Градиент функции f(x) = sin(x^2 + 1) при x=1.5")
print(f"Через .backward(): {grad_pytorch:.4f}")
print(f"Через torch.autograd.grad: {grad_autograd:.4f}")
print(f"Аналитически (cos(x^2+1)*2x): {grad_analytical:.4f}")
print("\n Задание 2 (homework_autograd.py) успешно выполнено!")