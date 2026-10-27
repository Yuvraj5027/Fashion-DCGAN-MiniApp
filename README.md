# Fashion Image Generator Using DCGAN

Mini application that trains a DCGAN on a small subset of Fashion-MNIST and generates new clothing-like images.

- `backend.py` - dataset loader, generator, discriminator, training loop, checkpoint saving
- `app.py` - Streamlit frontend (training controls, epoch display, generated-image gallery, loss record)
- `fashion_dcgan.ipynb` - notebook version with test cases

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
