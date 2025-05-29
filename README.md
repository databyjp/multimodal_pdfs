## Multimodal PDF search demo

### Set up the environment.

Recommended method:
- install `uv`
- `uv venv`
- `source .venv/bin/activate`
- `uv sync`

### Set up `.env` file

```
APP_WEAVIATE_CLOUD_URL=YOUR_WEAVIATE_URL
APP_WEAVIATE_CLOUD_APIKEY=YOUR_WEAVIATE_APIKEY
```

### Populate your database

Add your desired PDF to `data/src`

Run:

- `10_convert_pdf_to_imgs.py`
- `50_add_to_weaviate.py.py`

Then, run the app

`streamlit run app.py`
