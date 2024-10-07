pip show torch
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"

source env/bin/activate
accelerate launch ./prompt.py --input_csv /home/ali.lawati/mol-incontext/input/GCL-chebi-icl-new-2-100-epochs100.csv