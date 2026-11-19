# Обнаружение повторного JPEG-сжатия

Курсовая работа по дисциплине «Основы искусственных нейронных сетей», МГТУ им. Н. Э. Баумана. Автор — Гиричев М. С., группа ИУ7-76Б. Руководитель — Кудрявцев М. А.

Программа принимает фотографию в формате JPEG и относит её к одному из двух классов — однократное или двукратное сжатие. Она позволяет отобрать файлы с признаками повторного сохранения для дальнейшего анализа. Вместе с классом выводится оценка двукратного сжатия от 0 до 1. Для работы достаточно самого JPEG и весов из каталога `weights/`.

- [Заметки об эксперименте и выводы](docs/results.md)
- [Выполненный ноутбук с обучением и измерениями](train.ipynb)
- [Таблицы, предсказания и графики](research/)
- [Сравнение исходных и итоговых весов](research/protocol-comparison.csv) · [Архив исходного опыта на 20 эпох](archive/20-epochs/)
- [Пояснительная записка с приложением А](tex/report.pdf)
- Презентация к защите в [PDF](tex/presentation.pdf) и [PowerPoint](tex/presentation.pptx) · [Текст выступления по слайдам](docs/defense-speech.md)
- [Лицензия MIT](LICENSE)

## Запуск программы

Команды выполняются из корня проекта. В эксперименте использовался Python 3.12. Для классификации достаточно четырёх библиотек.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.5.5 Pillow==13.0.0 scipy==1.18.1 torch==2.15.0
python detector.py "/путь/к/фотографии.jpg"
```

Классификатор работает на CPU. На входе поддерживаются полутоновые и цветные JPEG с яркостной компонентой Y и 8-битной таблицей квантования. Минимальный размер — 256 × 256 пикселей. Из большего изображения берётся центральный фрагмент этого размера с границей на сетке блоков 8 × 8. Класс характеризует этот фрагмент. Повторное сохранение возможно и при обычной пересылке фотографии.

Для обработки каталога и его подкаталогов укажите путь к каталогу. Параметр `--csv` сохраняет путь, числовой класс (0 или 1), оценку двукратного сжатия и сообщение об ошибке чтения. По умолчанию программа загружает CNN с инициализацией 43 и 21-й эпохи. Параметр `--weights` позволяет выбрать другой файл. У итогового MLP с dropout 20 % выбраны инициализация 29 и 15-я эпоха. Итоговые веса лежат в `weights/60-epochs/`. Файлы непосредственно в `weights/` с именами вида `MLP-17.pt` сохранены для запуска исходных моделей.

```sh
python detector.py ./photos --csv predictions.csv
python detector.py photo.jpg --weights weights/60-epochs/MLP-29.pt
```

## Воспроизведение исследования

Обучение проходит локально на GPU Apple через PyTorch MPS. Для запуска готового классификатора загружать наборы данных не требуется. Для подготовки обучающей и тестовых выборок нужно около 30 ГБ свободного места.

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
curl -fL --retry 5 --retry-all-errors -C - 'https://huggingface.co/datasets/plok5308/djpeg_dataset/resolve/f07ee49be7c1072a1e0615852379d8fa5b05a09e/jpeg_data.tar.gz' -o data/jpeg_data.tar.gz
curl -fL --retry 5 'https://raw.githubusercontent.com/plok5308/DJPEG-torch/9cacb9671c7df85a2d0e006faff3c7d8cb656f26/qtables/q_table_list.mat' -o data/q_table_list.mat
curl -fL --retry 5 'https://codeload.github.com/girfa/ColorImageDatasets/tar.gz/cea0b27af75c2e9d4f966153bc6d7544c1a8df25' -o data/ucid-source.tar.gz
python prepare_data.py djpeg
python prepare_data.py ucid
python -m ipykernel install --prefix .venv --name jpeg-course --display-name "Python (JPEG)"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=jpeg-course --ExecutePreprocessor.timeout=1800 train.ipynb
```

В Jupyter можно открыть тот же ноутбук, выбрать ядро «Python (JPEG)» и выполнить все ячейки. Подготовка DJPEG сверяет SHA-256 исходного архива, извлекает готовые размеченные пары и вычисляет признаки. Для независимого опыта скрипт кодирует фрагменты TIFF-фотографий UCID с заданными параметрами JPEG. Обе сети обучаются с пределом в 60 эпох. После пяти эпох без снижения валидационной кросс-энтропии обучение останавливается. Ноутбук записывает веса в `weights/60-epochs/`, а результаты тестов на DJPEG и UCID и замеры времени — в `research/`. Исходные запуски и их результаты доступны в `archive/20-epochs/`. [Данные DJPEG](https://huggingface.co/datasets/plok5308/djpeg_dataset) · [Исходные фотографии UCID](https://github.com/girfa/ColorImageDatasets/tree/cea0b27af75c2e9d4f966153bc6d7544c1a8df25/UCID-1338)

## Пояснительная записка

```sh
./build.sh
```

Скрипт собирает `tex/report.pdf` с помощью Tectonic и совместимого Biber либо TeX Live с XeLaTeX и latexmk. Для явного выбора Biber предусмотрена переменная `BIBER_BIN`. Лист задания расположен после титульного листа. Методика, результаты и границы применимости моделей приведены в исследовательском разделе. [Образец оформления](https://github.com/MarkGiri/course-work-CG-BMSTU-5-sem/tree/main/tex) · [Образец листа задания](https://github.com/MarkGiri/course-work-DB-BMSTU-6-sem/tree/main/report/tex)
