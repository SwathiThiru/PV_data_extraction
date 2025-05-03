<h1>Extraction of Tabular data from PV module datasheets</h1>

The main goal of this was work was to recognize the structure of complex tables containing nested multi-level headers and merged cells to exsure accurate data extraction.

The proposed pipeline integrates Deep Learning (DL) models capable of Table Detection (TD) and
Table Structure Recognition (TSR) with optical character recognition (OCR) to transform PDF
datasheets into structured outputs. A transformer-based Detection Transformer (DETR) model
was initially trained to localize tables present in documents. This was followed by a second DETR
model that was trained to detect rows, columns, merged cells, multi-row/multi-column headers,
and other complex structures. An improved table structure refinement and canonicalization algorithm was also implemented to
process varying table layouts(vertical, horizontal and dual-axis tables). Post-processing techniques like Naive Bayes classifier and
regular expressions pattern matching and integration of LLMs enabled validation and extraction of relevant data from the
datasheets to aid PV module research.

<h2>Code Installation</h2>
Create a conda environment from the yml file and activate it as follows

```bash
conda env create -f environment-thesis.yml
conda activate tables-detr-thesis
```

<h2>Inference</h2>
To run the table detection model:

```bash
python src/inference_4thDecemberfinal.py --mode detect --detection_config_path src/detection_config.json --detection_model_path weights/detection20_finetume_with_words.pth --detection_device cpu --in_dir inferences/pdf  --out_dir inferences/detectionOutput --crop_padding 25 -p
```

To run the table structure recognition model:
```bash
python src/inference_4thDecemberfinal.py --mode recognize --structure_config_path src/latest_structure_config_old.json --model_dir inferences/models --structure_model_path weights/exponentialLR_recognition_with_cannonical_finetuned_v2_model_20.pth --structure_device cpu --in_dir inferences/detectionOutput  --out_dir inferences/recognitionOutput -c -z --crop_padding 20
```

To run the entire pipeline:
```bash
python src/inference_4thDecemberfinal.py --mode extract --detection_config_path src/detection_config.json --detection_model_path weights/detection_finetuned_HPO_negatives.pth --detection_device cpu --structure_config_path src/latest_structure_config_old.json --structure_model_path weights/exponentialLR_recognition_with_cannonical_finetuned_v2_model_20.pth --structure_device cpu --in_dir inferences/pdf --out_dir inferences/extractionOutput --model_dir inferences/models -c -z --crop_padding 20
```

<h2>Reproduce</h2>
My Master thesis work 'Thesis_report_Swathi_Thiruvengadam_5253272' documents the methodology, model selection and training overview snd finally presents the evaluation results of the pipeline along with future work.

Refer the Dockerfile and makefile to reproduce the results and see all the 
