# How to make everything work

1. Create an envinronment putting these commands on terminal:

- conda create --name minicondaenv python=3.8
- conda activate minihackenv

2. Install dependencies (mandatory only cmake) Ubuntu:


- sudo apt install build-essential libssl-dev
- sudo apt install cmake
- sudo apt install libbz2-dev
- sudo apt install libclang-dev
- sudo apt-get install bison
- sudo apt-get install flex

3. Install libraries:

- pip install nle
- pip install minihack
- pip install matplotlib notebook
- jupyter notebook


# Funziona!
- sudo apt update
- sudo apt install -y build-essential cmake g++ libbz2-dev liblzma-dev libncurses5-dev libssl-dev zlib1g-dev
- pip install --upgrade pip setuptools wheel
- sudo apt install ntp
- sudo service ntp restart
- pip install nle
- pip install git+https://github.com/facebookresearch/nle.git
- gcc --version
- g++ --version

- `nle` richiede generalmente una versione di `gcc` e `g++` >= 7. Se la tua versione è vecchia, aggiorna con:
  - sudo apt install gcc-9 g++-9
  - sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60
  - sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60

- pip install minihack
