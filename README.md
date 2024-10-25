# NavTarang: An efficient encoder-decoder architecture with top-down attention for speech separation

## Training and evaluation

### Inference with Pretrained Model
```python
import os
import torch
import look2hear.models
import torchaudio

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


mix, sr = torchaudio.load("audio_mix.wav")
transform = torchaudio.transforms.Resample(sr, 16_000)
mix = transform(mix)
mix = mix.view(1, 1, -1)
model = look2hear.models.BaseModel.cuda()
est_sources = model(mix.cuda())
torchaudio.save("audio1sep.wav", est_sources[:, 0, :].detach().cpu(), 16_000)
torchaudio.save("audio2sep.wav", est_sources[:, 1, :].detach().cpu(), 16_000)
```

## Results

A sample visual representation of our model:
## Home page
![Screenshot_25-10-2024_8525_doondigangadhar30 github io](https://github.com/user-attachments/assets/0490841b-e586-4d7b-b0b5-31b0c36e771d)
## Features page
![Screenshot_25-10-2024_8111_doondigangadhar30 github io](https://github.com/user-attachments/assets/6b2ba24b-9226-4f76-9bb3-69a2729cb017)
## Demo page
![Screenshot_25-10-2024_859_doondigangadhar30 github io](https://github.com/user-attachments/assets/f23a3882-935f-4483-82e1-d635e15dffcc)

## Links

- [Live Site URL]( https://doondigangadhar30.github.io/NavTarang/)
 ## Dataset Download Link: 
- [Google Driver](https://drive.google.com/file/d/1dCWD5OIGcj43qTidmU18unoaqo_6QetW/view?usp=sharing)
