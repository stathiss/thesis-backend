# thesis-stathis

## Requirements

- Python 2.7 (https://www.python.org/).


## Development

### Preparation
 
Create a virtual environment for Python 2.7 (needs to be done only once):

    virtualenv --python=python2.7 venv

Use the virtual environment:

    source venv/bin/activate

Install pip version 18.0:

    pip install pip==18.0

Install dependencies using pip:

    pip install -r requirements.txt
    

Export flask develop variable:

    export FLASK_ENV=development

Edit `config.py` file according to your tokens, databases, etc.


### Linter:
    
Run the linter that checks for violations of the PEP8 coding style.
    
    pycodestyle --max-line-length 120 --ignore=E402,E121,E123,E126,E226,E24,E704,E722,W503 sources/code/ sources/loaders/ sources/preprocessing/

Or you can set a custom command `code_style_check`:
    
    alias code_style_check='pycodestyle --max-line-length 120 --ignore=E402,E121,E123,E126,E226,E24,E704,E722,W503 sources/code/ sources/loaders/ sources/preprocessing/'


## Deploy

Run server:

    flask run

You should be able to see server running at (http://localhost:5000/)


