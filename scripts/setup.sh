
python3 -m venv venv
source venv/bin/activate

pip install -r requirements_dev.txt
python3 -m pip install -e .

# for requirements_dev.txt: mkdocstrings and mkdocstrings.python