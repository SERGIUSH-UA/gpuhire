# Обліковки й файли конфігурації

Тека конфігурації: `%LOCALAPPDATA%\gpurunner` (Windows), `~/.config/gpurunner`
(Linux, macOS); перекривається `GPURUNNER_CONFIG_DIR`. Змінні оточення завжди
сильніші за файли.

Більшість файлів руками писати не треба — їх кладе `gpurunner auth`:

| бекенд | команда | що з'являється |
|---|---|---|
| Kaggle | — | береться `~/.kaggle/kaggle.json` або `KAGGLE_USERNAME` + `KAGGLE_KEY` |
| Modal | `modal token new` | `~/.modal.toml` |
| Colab | `gpurunner auth google` | `google_client_secret.json` (OAuth-клієнт типу Desktop, завантажений із Google Cloud Console) і `google_token.json` |
| Vast.ai | `gpurunner auth vast` | `vast_api_key`; приватний SSH-ключ — `GPURUNNER_VAST_SSH_KEY` |
| Lightning | `gpurunner auth lightning --user-id … --api-key …` | `lightning.json` (`user_id`, `api_key`, `teamspace`) |
| Beam | `gpurunner auth beam --token …` | `beam.json`; або `~/.beam/config.ini` від `beam configure` |
| Saturn | `gpurunner auth saturn --url … --token …` | `saturn.json` (`url`, `token`) |

`gpurunner auth <бекенд> --verify` робить пробний запит і каже, чого бракує.

## R2 (доставка кадрів на орендований бокс)

Єдиний файл, який пишеться руками: `r2.json` за зразком
[`r2.example.json`](r2.example.json). Ті самі назви працюють як змінні оточення.
Ключі R2 на бокс не їдуть — туди потрапляють лише тимчасові підписані
посилання.

🔴 Обмежте токен R2 одним бакетом. gpurunner пише лише в бакет із `R2_BUCKET`,
але токен на весь акаунт відкриває й решту.

## Ліміти витрат

| змінна | зміст |
|---|---|
| `GPURUNNER_BEAM_MONTHLY_BUDGET` | місячна стеля Beam, $ |
| `GPURUNNER_BEAM_MAX_RUN_COST` | стеля одного прогону Beam, $ (разово — `run --allow-cost N`) |
| `GPURUNNER_MODAL_MONTHLY_CREDIT` | місячний кредит Modal для показу залишку |
