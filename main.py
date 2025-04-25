import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1) Определяем модели
def logistic(t, P, r, K):
    """Оригинальная логистическая модель Ферхюльста."""
    return r * P * (1 - P / K)

def allee(t, P, r, K, A):
    """Модель с эффектом Олли."""
    return r * P * (1 - P / K) * (P/A - 1)

# 2) Задаём параметры 
r  = 0.15    # скорость роста (в год)
K  = 800     # ёмкость среды (особей)
A  = 120   # порог эффекта Олли (особей)
P0 = 100     # начальная численность (особей)

# 3) Временной интервал решения
t_start, t_end = 0, 40        # годы
t_eval = np.linspace(t_start, t_end, 500)

# 4) Решаем ОДУ численно
sol_logistic = solve_ivp(
    fun=logistic,
    t_span=(t_start, t_end),
    y0=[P0],
    args=(r, K),
    t_eval=t_eval
)

sol_allee = solve_ivp(
    fun=allee,
    t_span=(t_start, t_end),
    y0=[P0],
    args=(r, K, A),
    t_eval=t_eval
)

# 5) Строим график
plt.figure(figsize=(8,5))
plt.plot(sol_logistic.t, sol_logistic.y[0], label='Логистика без Олли', linewidth=2)
plt.plot(sol_allee.t,    sol_allee.y[0],    label='С эффектом Олли', linewidth=2)
plt.title('Сравнение моделей Ферхюльста\nбез эффекта Олли и с эффектом Олли')
plt.xlabel('Время (лет)')
plt.ylabel('Численность популяции P(t)')

param_text = f'Параметры:\nr = {r}\nK = {K}\nl = {l}\nP0 = {P0}'
plt.text(1.005, 0.2, param_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
