### setting
- pip3 install torch torchvision torchaudio
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

- pip install -U accelerate==0.29.3 peft==0.10.0 bitsandbytes==0.43.1 transformers==4.40.1 trl==0.8.6 datasets==2.19.0

### check GPU power

- torch.cuda.get_device_capability()[0] >= 8