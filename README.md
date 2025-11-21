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
### **Third-Party Software Used**
- **AntiBERTy** | ([GitHub](https://github.com/jeffreyruffolo/AntiBERTy)) | [MIT](https://opensource.org/license/mit)
- **ESM-2** | ([GitHub](https://github.com/facebookresearch/esm)) | ([version](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)) | [MIT](https://opensource.org/license/mit)
- **ProtTrans** | ([GitHub](https://github.com/agemagician/ProtTrans)) | ([version](https://zenodo.org/record/4644188)) | [MIT](https://opensource.org/license/mit)
- **NetSurfP-3.0** | ([web-server](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/))
- **BLAST+ 2.12.0** | ([Download](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/)) | [NCBI](https://blast.ncbi.nlm.nih.gov/Blast.cgi)

  
The dependency versions of the third-party software are as follows:
```
antiberty==0.1.1
fair-esm==2.0.0
transformers==4.34.1 (for ProtTrans)
```
Please follow the instructions in the corresponding GitHub repository to install it into your environment, download the specified version of the weight files.

## **3.📦Feature Generation**

- First, taking the S4169 dataset as example, open the **`./data/utils/S4169_config.json`** file and modify the corresponding **path parameters** to your own paths (the same applies to the other datasets. The **`M1101_config.json`** configuration file and features are shared between the **M1101** and **S645** datasets).
- Then, run the following script to extract features from `.csv` files.
```
python ./data/utils/run_all_features.py --config_file S4169_config.json
```
- **One hot**, **Physicochemical properties** and **BLOSUM62** will be automatically generated during training.
 
- Please visit [NetSurfP-3.0 online server](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/) for **RASA** generation.


## **4.🚀Training**

- Run the following script to train the **S4139** model, the same applies to other datasets.
- (Note: Make sure to check any paths that may be involved in the file.)
```
python ./main/S4169/train/train_S4169.py
```
## **5.🎯Predict**
- Run the following script to predict the **S4139** model, the same applies to other datasets.
```
python ./main/S4169/predict/predict_S4169.py
```
- [The pretrained model is provided.](https://github.com/iobio-zjut/SAMAffinity/releases/tag/v1.0)

