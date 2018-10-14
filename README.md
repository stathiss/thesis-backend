# thesis-stathis

## Requirements

- Python 2.7 (https://www.python.org/).

## Development

### Preparation
 
Create a virtual environment for Python 2.7 (needs to be done only once):

    virtualenv --python=python2.7 venv

(The commands that follow are all included in `init.sh` script)

Use the virtual environment:

    source venv/bin/activate

Install pip version 18.0:

    pip install pip==18.0

Install dependencies using pip:

    pip install -r requirements.txt
    

Run script:

    source init.sh


### Linter:
    
Run the linter that checks for violations of the PEP8 coding style.
    
    pycodestyle --max-line-length 120 --ignore=E402,E121,E123,E126,E226,E24,E704,W503 sources/code/ sources/loaders/ sources/preprocessing/

Or you can set a custom command `code_style_check`:
    
    alias code_style_check='pycodestyle --max-line-length 120 --ignore=E402,E121,E123,E126,E226,E24,E704,W503 sources/code/ sources/loaders/ sources/preprocessing/'


## Implementation

To do:

    to do


