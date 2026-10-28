# thompson_sample setup

Requires Python 3.11 or newer (Pillow 13).

- `pyenv virtualenv 3.11 thompson_sample`
- `pip install -r requirements.txt`
- `python run_gaussian.py`

<img width="785" alt="image" src="https://github.com/user-attachments/assets/b519fdda-fa43-4e4e-8076-faeb6e04b665">

Run the dependency smoke test without opening a window:

```sh
MPLBACKEND=Agg python -m unittest discover -s tests -v
black --check app tests run_gaussian.py
```
