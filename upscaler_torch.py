import torch


"""
Математика апскейлинга для одного канала цвета с использованием градиентов работает через построение бикубического сплайна на единичном квадрате между четырьмя соседними пикселями.

## Постановка задачи

Для интерполяции значения в произвольной точке (x,y) внутри единичного квадрата [0,1]×[0,1] используется полином третьей степени:

```
p(x,y) = Σᵢ₌₀³ Σⱼ₌₀³ aᵢⱼxⁱyʲ
```

Коэффициенты aᵢⱼ образуют матрицу A ∈ ℝ⁴ˣ⁴.

## Граничные условия

В углах единичного квадрата известны:
- Значения функции: f(0,0), f(1,0), f(0,1), f(1,1)
- Градиенты по x: fₓ(0,0), fₓ(1,0), fₓ(0,1), fₓ(1,1)
- Градиенты по y: fᵧ(0,0), fᵧ(1,0), fᵧ(0,1), fᵧ(1,1)
- Смешанные производные: fₓᵧ(0,0), fₓᵧ(1,0), fₓᵧ(0,1), fₓᵧ(1,1)

Эти 16 значений формируют матрицу F:

```
F = [f(0,0)   f(0,1)   fᵧ(0,0)  fᵧ(0,1)]
    [f(1,0)   f(1,1)   fᵧ(1,0)  fᵧ(1,1)]
    [fₓ(0,0)  fₓ(0,1)  fₓᵧ(0,0) fₓᵧ(0,1)]
    [fₓ(1,0)  fₓ(1,1)  fₓᵧ(1,0) fₓᵧ(1,1)]
```

## Вычисление коэффициентов

Коэффициенты A находятся через решение системы:

```
A = C⁻¹F(Cᵀ)⁻¹
```

где C - матрица базисных функций кубического сплайна:

```
C = [1  0  0  0]
    [1  1  1  1]
    [0  1  0  0]
    [0  1  2  3]
```

Эта матрица получается из условий на кубическую функцию f(x) = a₀ + a₁x + a₂x² + a₃x³ и её производную в точках 0 и 1.

## Интерполяция

После вычисления A, значение в произвольной точке (x,y) ∈ [0,1]×[0,1]:

```
p(x,y) = [1 x x² x³] · A · [1 y y² y³]ᵀ
```

## Использование аналитических градиентов

В контексте 3D Gaussian Splatting градиенты вычисляются аналитически через альфа-блендинг:

```
∂I/∂x = Σᵢ₌₁ᴺ cᵢ(∂Tᵢ/∂x·αᵢ + Tᵢ·∂αᵢ/∂x)
```

где:
- cᵢ - цвет i-го гауссиана
- Tᵢ - накопленная прозрачность до i-го гауссиана
- αᵢ = σᵢ·exp(gᵢ(x,y)) - непрозрачность i-го гауссиана
- gᵢ(x,y) = -dᵢᵀΣᵢ⁻¹dᵢ - квадратичная форма от расстояния до центра

Производные αᵢ по координатам:

```
∂αᵢ/∂x = αᵢ · ∂gᵢ/∂x = αᵢ · 2(Σᵢ⁻¹dᵢ)ₓ
```

Эти аналитические градиенты используются напрямую в матрице F вместо конечно-разностных аппроксимаций, что даёт более точную интерполяцию, особенно в областях с резкими изменениями сигнала.
"""




def bicubic_spline_upscale_single_channel(render, dx, dy, dxy, new_h, new_w):
    """
    Бикубическая сплайн-интерполяция для одного канала
    render, dx, dy, dxy: тензоры формы [h, w]
    """
    h, w = render.shape
    device = render.device

    # Константная матрица C и её обратная
    C = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 2, 3]
    ], dtype=torch.float32, device=device)

    C_inv = torch.inverse(C)

    # Создаём сетку координат
    y_coords = torch.linspace(0, h - 1, new_h, device=device)
    x_coords = torch.linspace(0, w - 1, new_w, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Индексы и веса
    y0 = torch.floor(grid_y).long()
    x0 = torch.floor(grid_x).long()
    y1 = torch.clamp(y0 + 1, 0, h - 1)
    x1 = torch.clamp(x0 + 1, 0, w - 1)

    ty = (grid_y - y0.float()).unsqueeze(-1)  # [new_h, new_w, 1]
    tx = (grid_x - x0.float()).unsqueeze(-1)  # [new_h, new_w, 1]

    # Клипируем индексы
    y0 = torch.clamp(y0, 0, h - 1)
    x0 = torch.clamp(x0, 0, w - 1)

    # Собираем значения в углах для всех позиций
    f00 = render[y0, x0]  # [new_h, new_w]
    f01 = render[y0, x1]
    f10 = render[y1, x0]
    f11 = render[y1, x1]

    fx00 = dx[y0, x0]
    fx01 = dx[y0, x1]
    fx10 = dx[y1, x0]
    fx11 = dx[y1, x1]

    fy00 = dy[y0, x0]
    fy01 = dy[y0, x1]
    fy10 = dy[y1, x0]
    fy11 = dy[y1, x1]

    fxy00 = dxy[y0, x0]
    fxy01 = dxy[y0, x1]
    fxy10 = dxy[y1, x0]
    fxy11 = dxy[y1, x1]

    # Формируем матрицу F для каждой позиции
    # [new_h, new_w, 4, 4]
    # Матрица F согласно математическому описанию:
    # F[0,:] = [f(0,0), f(0,1), fy(0,0), fy(0,1)] - значения и производные по y при x=0
    # F[1,:] = [f(1,0), f(1,1), fy(1,0), fy(1,1)] - значения и производные по y при x=1
    # F[2,:] = [fx(0,0), fx(0,1), fxy(0,0), fxy(0,1)] - производные по x и смешанные при x=0
    # F[3,:] = [fx(1,0), fx(1,1), fxy(1,0), fxy(1,1)] - производные по x и смешанные при x=1
    F = torch.stack([
        torch.stack([f00, f10, fy00, fy10], dim=-1),  # x=0
        torch.stack([f01, f11, fy01, fy11], dim=-1),  # x=1
        torch.stack([fx00, fx10, fxy00, fxy10], dim=-1),  # производные по x при x=0
        torch.stack([fx01, fx11, fxy01, fxy11], dim=-1)   # производные по x при x=1
    ], dim=-2)

    # Вычисляем коэффициенты A
    F_reshaped = F.reshape(-1, 4, 4)
    A = torch.matmul(torch.matmul(C_inv, F_reshaped), C_inv.t())
    A = A.reshape(new_h, new_w, 4, 4)

    # Полиномиальные базисные функции
    ones = torch.ones_like(tx)
    px = torch.stack([ones, tx, tx ** 2, tx ** 3], dim=-1)  # [new_h, new_w, 1, 4]
    py = torch.stack([ones, ty, ty ** 2, ty ** 3], dim=-1)  # [new_h, new_w, 1, 4]

    # Вычисляем значения полинома
    # p(x,y) = px^T * A * py
    # px: [new_h, new_w, 1, 4] -> squeeze to [new_h, new_w, 4]
    # A: [new_h, new_w, 4, 4]
    px_squeezed = px.squeeze(-2)  # [new_h, new_w, 4]
    py_squeezed = py.squeeze(-2)  # [new_h, new_w, 4]

    # Perform batched matrix multiplication
    temp = torch.einsum('hwi,hwij->hwj', px_squeezed, A)  # [new_h, new_w, 4]
    upscaled = torch.einsum('hwi,hwi->hw', temp, py_squeezed)  # [new_h, new_w]

    return upscaled
