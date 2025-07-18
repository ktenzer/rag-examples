# rag-examples

## Install Dependencies
```bash
$ poetry install
```

## Configuration
```bash
$ cp .env.example .env
```

Setup environment depending on using openai or webAI

## OpenAI
```bash
$ poetry run python chatgpt.py
```

## webAI
Ensure API and LLM Chat elements are running on correct port and with the right api key

```bash
$ poetry run python webai.py
```


