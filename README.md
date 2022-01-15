# Access to Dataset
The training dataset is created based on the Sentinel-2 Satellite Image of the Gaza Strip. The four steps in my script explain into detail how to build the training dataset by few processing steps of the satellite image. If you want to follow all steps from the scratch, you should download the satelite image as explained below. But if you just want to test the script on the already extracted traing data, you can skip the first four steps in the script and start directly from Step 5. The training dataset is stored in the file `data.rda`. You can find this data file in the `data` folder.

## Down;load the Satellite Image
1) Download the Satellite Sentinel-2 Image from the following link:
   https://scihub.copernicus.eu/dhus/odata/v1/Products('cc447968-a42c-49b0-a4f1-08936b64ca6b')/$value
2) If username or password is required, please use the following credentials:
      username: malhessi
      password: Sally@2019!

3) A zipped file is downloaded with the name: "S2B_MSIL2A_20210525T081609_N0300_R121_T36RXV_20210525T105614.zip", Extract it in this folder `S2B`.
4) After Extraction, a folder is created with extension *.SAFE. its name is "S2B_MSIL2A_20210525T081609_N0300_R121_T36RXV_20210525T105614.SAFE".
5) You are done. The script is now able to read the image and process it.
