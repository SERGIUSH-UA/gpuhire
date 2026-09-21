# gpurunner

Один CLI для GPU-задач на чужому залізі: подати, стежити, забрати результат,
не переплатити. Сім бекендів за одним інтерфейсом — Kaggle, Modal, Google Colab,
Vast.ai, Lightning, Beam, Saturn Cloud.

> **In English.** `gpurunner` is a command-line tool that submits GPU jobs (OCR,
> handwriting recognition, model training) to Kaggle, Modal, Colab, Vast.ai,
> Lightning, Beam or Saturn Cloud through one interface, tracks them in a local
> manifest, fetches outputs and keeps an eye on spending. Documentation and CLI
> messages are in Ukrainian. It works on its own and also serves as the GPU
> rental module of [nyshporka](https://github.com/SERGIUSH-UA/nyshporka).

Статус: альфа (0.2). Команди й машинний вивід, на які спираються інші програми,
описано в [`docs/contract.md`](docs/contract.md) і в межах мінорної версії не
міняються; решта ще рухається.

## 1. Навіщо

- Той самий job іде на безкоштовну квоту Kaggle, на Modal із посекундним
  білінгом або на орендований бокс Vast — міняється лише `-b`.
- Кожен прогін лишає запис у локальному маніфесті: що подано, куди, з якими
  параметрами, чим закінчилось і чому впало.
- Гроші видно до старту: прорахунок вартості, стеля витрат, залишки по бекендах
  (`gpurunner balance`), і що зараз тарифікується (`gpurunner burn`).
- Для довгих заходів на Vast є наглядач (`gpurunner htr supervise`): орендує
  бокс, стежить за темпом, доганяє пропуски, звіряє повноту й гасить оренду сам.
- Бакет S3 для цього не обов'язковий: `htr plan --transport box` везе кадри й
  ваги на саму машину по `scp` і піднімає на ній крихітний склад, з якого їх
  бере робота. Чекпоінти наглядач забирає додому сам — машина може вмерти.

## 2. Установлення

Потрібен Python 3.11+.

⚠ На PyPI пакет зветься **`gpuhire`**, а команда в терміналі — `gpurunner`.
Ім'я `gpurunner` на PyPI зайняте схожим проєктом (`gpu-runner`), і майданчик
не дав його взяти; перейменовувати команду заради цього не стали.

```
pip install gpuhire                 # ядро + Kaggle
pip install "gpuhire[modal]"        # + Modal
pip install "gpuhire[vast,r2]"      # + Vast.ai і доставка даних через R2
pip install "gpuhire[all]"          # modal, colab, vast, web, r2
```

`lightning`, `beam`, `saturn` ставляться поіменно: їхні SDK важкі й потрібні
одиницям. Як модуль Нишпорки: `pip install "gpuhire[vast,r2]"` поруч із
нею — Нишпорка знайде плагін сама.

Поруч із Нишпоркою gpurunner стає її **бекендом оренди**: оголошує entry point
`nyshporka.cloud`, і `nysh cloud` сам орендує бокс на Vast.ai, читає справу по
SSH, забирає результат і гасить оренду. Від людини — лише ключ API
(`gpurunner auth vast --key <КЛЮЧ>`) і баланс; SSH-ключ генерується сам, R2 не
потрібен. Докладно — [`docs/nyshporka.md`](docs/nyshporka.md).

З вихідного коду:

```
git clone https://github.com/SERGIUSH-UA/gpurunner && cd gpurunner
uv sync --extra all
uv run gpurunner --version
```

## 3. Перший прогін

```
gpurunner auth kaggle --verify        # обліковка на місці?
gpurunner ls                          # які є job'и й бекенди
gpurunner run net-probe -b kaggle     # найдешевша перевірка: чи є в кернелі мережа
gpurunner watch <handle>              # чекати до кінця
gpurunner fetch <handle> -o out/      # забрати вихід
```

`--dry-run` перевіряє параметри й показує код, який поїде на бекенд, нічого не
подаючи. Параметри job'а — `-p ключ=значення` (значення розбирається як JSON,
якщо виходить).

🔴 **Vast тарифікує погодинно й не спиняється після завершення job'а.** Після
`fetch` обов'язково `gpurunner cancel <handle>`; що зараз горить, показує
`gpurunner burn`.

## 4. Бекенди

| бекенд | extra | обліковка | особливість |
|---|---|---|---|
| `kaggle` | — | `~/.kaggle/kaggle.json` або `KAGGLE_USERNAME` + `KAGGLE_KEY` | безкоштовна тижнева квота; вхід — Kaggle Dataset |
| `modal` | `modal` | `~/.modal.toml` (`modal token new`) | посекундний білінг; дані — у томі Modal; локальний Python має бути 3.12 — як в образі |
| `colab` | `colab` | OAuth-клієнт Google у теці конфігурації | нотбук треба запустити руками: Runtime → Run all |
| `vast` | `vast` | `VAST_API_KEY` або файл `vast_api_key` | оренда боксу по SSH; тарифікація до `cancel` |
| `lightning` | `lightning` | `LIGHTNING_USER_ID`, `LIGHTNING_API_KEY` | Studio + teamspace |
| `beam` | `beam` | `BEAM_TOKEN` або `~/.beam/config.ini` | стеля витрат на прогін і на місяць |
| `saturn` | `saturn` | `SATURN_TOKEN`, `SATURN_BASE_URL` | типи інстансів — `gpurunner saturn sizes` |

`gpurunner auth <бекенд>` показує, де шукається обліковка й чого бракує;
`--verify` робить пробний запит. Тека конфігурації — `%LOCALAPPDATA%\gpurunner`
(Windows) або `~/.config/gpurunner`; перекривається `GPURUNNER_CONFIG_DIR`.
Приклади файлів — у [`docs/config/`](docs/config/).

## 5. Job'и

Перелік з описами й бекендами — `gpurunner ls`; готові набори параметрів із
прорахунком вартості — `gpurunner recipes <job>`.

Загальні:

| job | що робить |
|---|---|
| `net-probe` | чи є у віддаленому кернелі мережа |
| `paddleocr` | PaddleOCR PP-OCRv5 по переліку PDF за посиланнями → txt і json по сторінках |
| `vit_classifier`, `crop_verifier`, `dino_surname_verifier` | дотренування бінарного класифікатора зображень (ViT / DINOv2) і скоринг |
| `yolo_spotter` | дотренування YOLO на пошук одного рукописного слова |
| `churro`, `rukopys_ocr` | сторінкове читання історичних документів VLM-моделями |
| `kraken_lines`, `trocr_lines` | сегментація сторінок на рядки й читання рядків |
| `kraken_train`, `parseq_train` | трен моделей читання рядків (kraken, PARSeq) |
| `htr_eval`, `htr_lines_eval`, `htr_page_bench`, `paddle_page_bench`, `spotter_page_bench` | бенчмарки читання рукопису |

Потребує пакета `nyshporka`:

| job | що робить |
|---|---|
| `htr_case` | прогін архівної справи раннером Нишпорки на орендованій карті; раннер і моделі приїжджають архівом ассетів, gpurunner доставляє їх на бокс і стежить |

Команди `gpurunner htr …` (план, передполіт, наглядач, забір чекпоінтів,
калібрування) обслуговують саме `htr_case`.

## 6. Команди

```
run · status · watch · logs · fetch · cancel · sweep      життєвий цикл прогону
ls · recipes · balance · burn · reconcile · dash           огляд, гроші, звірка
auth <бекенд>                                              обліковки
dataset push|files                                         Kaggle-датасети
drive push|files                                           вхідні дані Colab у Google Drive
vast offers|instances · saturn sizes                       ринок і типи машин
boxes ls|explain|ban|star|prune|forget                     реєстр орендних боксів
bg start|status|stop                                       довгі задачі у фоні
htr plan|preflight|supervise|state|stop|quiesce|fetch-ckpt|calibrate|append
```

`gpurunner <команда> --help` — повний перелік прапорців.

## 7. Реєстр боксів

Кожна оренда на Vast лишає рядок «обіцяне поруч із виміряним»: яке залізо
заявив хост і який темп воно дало. З цього складається вердикт машини, і
наступний пошук оферти погані машини оминає, а добрі шукає адресно.

Реєстр — дані користувача, а не пакета: у ньому ідентифікатори машин, ціни й
назви власних прогонів. Лежить у `GPURUNNER_REPO_DATA_DIR`, без змінної — у
`<тека даних>/registry`. Змінну варто спрямувати в теку власного
git-репозиторію: тоді в дифі видно, коли й чому бокс став поганим.

## 8. Для програм

Інші програми кличуть gpurunner окремим процесом і читають `--json`: останній
рядок stdout, що починається з `{`, — один об'єкт. Форма відповідей, перелік
стабільних команд, схема `plan.json` і змінні оточення —
[`docs/contract.md`](docs/contract.md).

Другий шлях — плагін у тому самому інтерпретаторі: клас
`gpurunner.plugins.nyshporka_cloud:VastRent` (entry point `nyshporka.cloud`)
орендує й гасить бокс для конвеєра Нишпорки. Методи, форма `Box.meta` і
необов'язкові `status / login / estimate` — [§6 контракту](docs/contract.md#6-плагін-nyshporkacloud).

## 9. Розробка

```
uv sync --extra all
uv run pytest
uv run ruff check .
uv run mypy src
python tools/scan_private.py          # ворота проти приватних даних
```

Тести не ходять у мережу й не потребують обліковок. Порядок внесення змін і
чекліст релізу — [`CONTRIBUTING.md`](CONTRIBUTING.md); про вразливості —
[`SECURITY.md`](SECURITY.md).

## 10. Ліцензія

MIT — див. [`LICENSE`](LICENSE).
