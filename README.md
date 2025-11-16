# SAMAffinity

## 📬 Contact (Supervisor)

**Prof. Guijun Zhang**  
College of Information Engineering  
Zhejiang University of Technology, Hangzhou 310023, China  
✉️ Email: [zgj@zjut.edu.cn](mailto:zgj@zjut.edu.cn)

## 🌟**Overall workflow for the SAMAffinity**🌟
![SAMAffinity pipeline](pipeline.png)

## **1.🛠Download SAMAffinity package**

```
git clone --recursive https://github.com/iobio-zjut/SAMAffinity 
```
- **Download [the pretrained model weights](https://github.com/iobio-zjut/SAMAffinity/releases/tag/v1.0) used in SAMAffinity.**

- **Add the download file** to `/SAMAffinity/save_models`.

## **2.🔥Installation**

### **Third-Party Software Used**
- **AntiBERTy** | ([GitHub](https://github.com/jeffreyruffolo/AntiBERTy)) | [MIT](https://opensource.org/license/mit)
- **ESM-2** | ([GitHub](https://github.com/facebookresearch/esm)) | ([version](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)) | [MIT](https://opensource.org/license/mit)
- **ProtTrans** | ([GitHub](https://github.com/agemagician/ProtTrans)) | ([version](https://zenodo.org/record/4644188)) | [MIT](https://opensource.org/license/mit)

### **Create a new conda environment and update**

``` 
conda create -n SAMAffinity python=3.7
conda activate SAMAffinity
```

### **Install dependencies**

```
torch==1.13.0
torchaudio==0.13.0
torchvision==0.14.0
pandas==1.3.5
python==3.7.12
numpy==1.18.5
scikit-learn==1.0.2
scipy==1.7.3
```

## **3.📦Feature Generation**

- First, run the following script to convert '.csv' to '.fasta' file.
```
python ./data/utils/process_csv_to_fasta.py
```

- Then, run the following script to extract features from '.fasta' files.
```
python ./data/utils/AntiBERTy_embedding_generate.py
python ./data/utils/ESM2_embedding_generate.py
python ./data/utils/Prott5_embedding_generate.py
python ./data/utils/PSSM_generate.py
```
- **One hot**, **Physicochemical properties** and **BLOSUM62** will be automatically generated during training.
 
- Please visit [NetSurfP-3.0 online server](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/) for RASA generation.


## **4.🚀Training**

- Run the following script to train the **S1131** model, the same applies to other datasets.
```
python ./main/S1131/train/train_S1131.py
```

## **5.🎯Predict**
- Run the following script to predict the **S1131** model, the same applies to other datasets
```
python ./main/S1131/predict/predict_S1131.py
```
- [The pretrained model is provided.](https://github.com/iobio-zjut/SAMAffinity/releases/tag/v1.0)
