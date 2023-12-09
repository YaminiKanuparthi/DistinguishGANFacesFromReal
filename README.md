## Distinguish Real and Fake Images

We are analyzing state of the art models to distinguish Real and Fake Human Faces. The StyleGan2 and other AI generated Images are used for the analysis.

### Datasets Used:

*   Real:
    1.  [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) used as [archive.zip](https://drive.google.com/file/d/1_UYjNhjsdVxoOy0rmpTyGbOj-xUiVFwQ/view?usp=sharing)
    2.  [Celeb A](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
    3.  [Flickr Images Dataset](https://github.com/NVlabs/ffhq-dataset)
*   Fake:
    1.  [StyleGan2-ADA Pytorch](https://github.com/HarshitaDPoojary/DistinguishGANFacesFromReal/blob/main/Dataset%20Preparation/SG2_ADA_PyTorch.ipynb) taken from [StyleGan2 Resource](https://ckeditor.com/docs/ckeditor5/latest/features/autoformat.html)
    2.  [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
    3.  [Thispersondoesnotexist](https://www.kaggle.com/datasets/omjannu/thispersondoesnotexist)

All images combined are used as [Final Dataset](https://drive.google.com/file/d/1x9eB7Bk2jiekJT85ALIocStIDvWRi-ur/view?usp=sharing)

### Models Under Analysis

1.  **ViT**:
    *   _**Datasets Used**_: 140k Real vs Fake
    *   _**Implementation**_:
        *   [HugginFace ViT](https://huggingface.co/docs/transformers/model_doc/vit#vision-transformer-vit)
        *   Implemented in colab
        *   Change the path to directories
        *   Train the model
        *   Final Model Checkpoints: [vit_model.pth](https://drive.google.com/file/d/17GV8Eg91kTeDVhAx800X_wIL6VXbVae8/view?usp=sharing), [vit_model_finetuned.pth](https://drive.google.com/file/d/1-xnZuB5zX-IxCGchqFnQuTxZD8s067tw/view?usp=sharing)
2.  **CvT**:
    *   _**Datasets Used**_: 140k Real vs Fake
    *   _**Implementation**_:
        *   [HugginFace CvT](https://huggingface.co/docs/transformers/model_doc/cvt#convolutional-vision-transformer-cvt)
        *   Implemented in colab
        *   Change the path to directories
        *   Train the models
        *   Final Model Checkpoints: [CNNViT_model.pth](https://drive.google.com/file/d/1Bm_UxzUQKNQrHj3gMHEhKAqTkLuh2Wi1/view?usp=sharing)
3.  **LeViT**:
    *   _**Datasets Used**_: Combined (Final)
    *   _**Implementation**_:
        *   Models used
            1.  [LeViT](https://huggingface.co/docs/transformers/model_doc/levit#transformers.LevitForImageClassification)
                *   [Version 1](https://github.com/HarshitaDPoojary/DistinguishGANFacesFromReal/blob/main/Analysis/Transformer%20Analysis/LeVit.ipynb): Trained on 140k Real vs fake
                *   [Version 2](https://github.com/HarshitaDPoojary/DistinguishGANFacesFromReal/blob/main/Analysis/Transformer%20Analysis/LeVit_scratch.ipynb): Trained on Combine Dataset
            2.  [LeViT(With Knowledge Distillation)](https://huggingface.co/docs/transformers/model_doc/levit#transformers.LevitForImageClassificationWithTeacher)
        *   Since the model is lightweight the checkpoints are in this [repo](https://github.com/HarshitaDPoojary/DistinguishGANFacesFromReal/tree/main/Analysis/Transformer%20Analysis/models).
4.  **DCT/DFT**:
    *   _**Datasets Used**_: 140k Real vs Fake
    *   _**Implementation**_:
        1.  DCT:
            *   Using [GANDCTAnalysis](https://github.com/RUB-SysSec/GANDCTAnalysis/tree/master)
            *   Used DCT conversion
                *   `python3 .\prepare_dataset.py "C:\Users\dmpoo\OneDrive\Desktop\Harshita\realvsfake_merged_cropped" -lnc tfrecords`
            *   Using models: resnet
                *   Train:
                    *   `python ./classifer.py train resnet /content/drive/MyDrive/ML\ Project\ -\ Dump/GANDCTAnalysis/Dataset/tfrecords/realvsfake_merged_cropped_color_raw_normalized_train_tf/data.tfrecords /content/drive/MyDrive//ML\ Project\ -\ Dump/GANDCTAnalysis/Dataset/tfrecords/realvsfake_merged_cropped_color_raw_normalized_val_tf/data.tfrecords -b 32 -e 100 --l2 0.01 --classes 2 --image_size 128`
                *   Test:
                    *   `python ./classifer.py test /content/drive/MyDrive/GANDCTAnalysis/final_models/resnet_2023-11-28-22-01-11_batch_32_learning_rate_0.001/saved_modeel.pb content/drive/MyDrive/ML\ Project\ -\ Dump/GANDCTAnalysis/Dataset/tfrecords_dct/realvsfake_merged_cropped_color_dct_log_scaled_normalized_test_tf/data.tfrecords -b 32 --image_size 128`
            * Model Checkpoints: [Google Drive](https://drive.google.com/drive/folders/1-OHov_GUSMN1u-R-kou16CxjAHFReuOx?usp=sharing)
        2.  DFT:
            *   inspired by [Unmasking DeepFake with simple Features](https://github.com/cc-hpc-itwm/DeepFakeDetection/tree/master)
            *   Extract the images
            *   Convert to DFT
            *   Convert to 1D spectrum

<img align="center" src="img\pipeline.png" width="1000"/>