# Satranc Tasi Siniflandirma (Chess Piece Classification)

Bu proje, derin ogrenme ve bilgisayarli goru teknikleri kullanarak satranc taslarini siniflandirmayi amaclamaktadir. CNN (Evrisimli Sinir Aglari) mimarisi kullanilarak gelistirilen model; sah, vezir, kale, fil, at ve piyon olmak uzere 6 farkli satranc tasini yuksek dogrulukla taniyabilmektedir.

## Proje Ozellikleri

- TensorFlow ve Keras kullanilarak olusturulmus ozgun CNN mimarisi
- Goruntu on isleme ve veri artirma (data augmentation) adimlari
- Model basarim metrikleri: Loss ve Accuracy grafikleri
- Test seti uzerinde yuksek dogruluk orani

## Kullanilan Teknolojiler

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## Kurulum ve Calistirma

1. Bu depoyu klonlayin:
   ```bash
      git clone https://github.com/BaranBugdaci/Satranc_Tasi_Siniflandirma.git
         ```
         2. Gerekli kutuphaneleri yukleyin:
            ```bash
               pip install tensorflow opencv-python numpy matplotlib jupyter
                  ```
                  3. Jupyter Notebook uygulamasini baslatin ve Chess.ipynb dosyasini calistirin:
                     ```bash
                        jupyter notebook
                           ```

                           ## Model Mimarisi

                           Model, goruntulerden ozellik cikarimi yapmak icin ardisik Conv2D ve MaxPooling2D katmanlarindan olusmaktadir. Asiri ogrenmeyi (overfitting) onlemek amaciyla Dropout katmani eklenmis ve son katmanda 6 sinifli softmax aktivasyon fonksiyonu kullanilmistir.
