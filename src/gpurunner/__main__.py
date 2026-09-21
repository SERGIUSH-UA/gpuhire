"""`python -m gpurunner` — останній запасний шлях відчеплених задач.

Задача планувальника не має PATH, і коли ні `gpurunner` у PATH, ні
`gpurunner.exe` поруч не знайдено, `detach.self_argv()` кличе саме так. Доти
цього модуля не було, і запасний шлях падав на «No module named gpurunner.__main__».
"""

from gpurunner.cli import app

if __name__ == "__main__":
    app()
