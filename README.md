# Лабораторные по персонализированному управлению скидками

Тема ВКР: «Разработка многоагентной системы персонализированного управления скидками в электронной коммерции».

Пять независимых папок сдачи используют общий подготовленный набор данных и функции метрик. Исходные Word-задания не изменены. Общий протокол и фактические отклонения находятся в `docs/superpowers/`.

## Готовые материалы

- [Лабораторная 1 — LLM](lab01_llm/report.docx)
- [Лабораторная 2 — мультимодальные модели](lab02_multimodal/report.docx)
- [Лабораторная 3 — MARL](lab03_marl/report.docx)
- [Лабораторная 4 — оптимизация](lab04_optimization/report.docx)
- [Лабораторная 5 — исследовательский проект](lab05_defense/report.docx)
- [Презентация: 14 слайдов](lab05_defense/defense.pptx), [сценарий защиты](lab05_defense/demo_scenario.md), [архив](lab05_defense/artifacts.zip).

Восемь исполняемых проверок и офлайн-демо проходят. Полная визуальная проверка страниц DOCX/PPTX недоступна без Word/LibreOffice; выполнены структурные проверки и просмотр исходных графиков. Журнал: `lab05_defense/results/verification.json`.

## Запуск в Windows PowerShell

Требуется Python 3.12; локальные эксперименты выполнены на 3.12.14. В команде создания окружения `python` должен указывать на Python 3.12.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m shared.data
.venv/Scripts/python.exe shared/download_models.py
.venv/Scripts/python.exe lab01_llm/run.py
.venv/Scripts/python.exe lab02_multimodal/run.py
.venv/Scripts/python.exe lab03_marl/run.py
.venv/Scripts/python.exe lab04_optimization/run.py
.venv/Scripts/python.exe shared/analysis.py
.venv/Scripts/python.exe lab05_defense/build.py
.venv/Scripts/python.exe lab05_defense/demo.py
.venv/Scripts/python.exe lab05_defense/build.py --archive-only
```

`download_models.py` использует системный `curl.exe` с проверкой HTTPS. Веса моделей требуют отдельного сетевого доступа и не входят в сдаваемый архив. Для предсказаний не используются платные API.

## Проверка

```powershell
.venv/Scripts/python.exe shared/test_shared.py
.venv/Scripts/python.exe shared/audit.py
.venv/Scripts/python.exe shared/test_protocol.py
.venv/Scripts/python.exe lab01_llm/test_lab.py
.venv/Scripts/python.exe lab02_multimodal/test_lab.py
.venv/Scripts/python.exe lab03_marl/test_lab.py
.venv/Scripts/python.exe lab04_optimization/test_lab.py
.venv/Scripts/python.exe lab05_defense/build.py --archive-only
.venv/Scripts/python.exe lab05_defense/test_lab.py
```

`test_lab.py` проверяет код и артефакты, не заменяя основной эксперимент. Если запуск прерван, результат остаётся неполным. Не удаляйте `config.json`, manifest данных и raw logs при воспроизведении.

## Интерпретация

Покупки и товарный каталог — реальные открытые данные Complete Journey. Реакция на новые скидки и себестоимость — параметры симуляции. Прибыль в симуляции не является оценкой причинного эффекта для действующего интернет-магазина. У текстового, числового и графического представлений общий первичный источник; независимые отзывы и фотографии не добавлялись.

Время разбито до обучения, покупатели дополнительного переноса исключены из train. Победители выбираются по validation. У каждой обучаемой компактной модели и MARL-алгоритма пять seed. Полные конфигурации, ошибки и отрицательные результаты сохраняются в `results/`.
# labs_4
