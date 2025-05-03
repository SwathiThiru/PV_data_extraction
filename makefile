.PHONY: help detection-finetune recognition-finetune inference-detection inference-recognition inference-extraction evaluation-detection evaluation-recognition evaluation-extraction evaluation-final

help:
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  finetune-detection: Code to finetune the table detection model. A custom PV module dataset was used to finetune the model. Download it from the github repository and update the data_root_dir path accordingly before running the code. Update --device if using GPUs. The trained model weights will be stored in the --model_save_dir folder path. The pre-trained and fintuned model weights are also already available at /app/weights"
	@echo "  finetune-recognition: Code to finetune the table structure recognition model. A custom PV module dataset was used to finetune the model. Download it from the github repository and update the data_root_dir path accordingly before running the code. Update --device if using GPUs. The trained model weights will be stored in the --model_save_dir folder path. The pre-trained and fintuned model weights are also already available at /app/weights"
	@echo "  inference-detection: Runs inference pipeline to detect table from PDF files. Outputs are stored in 'app/inferences/detectionOutput'"
	@echo "  inference-recognition: Run inference pipeline to recognize structure of tables. Outputs stored in 'app/inferences/recognitionOutput'"
	@echo "  inference-extraction: Runs the complete tabular data extraction pipeline. Outputs stored in 'app/inferences/extractionOutput'"
	@echo "  evaluation-detection: Evaluates the preformance of the detection model on the custom PV module dataset comprising of complex tables."
	@echo "  evaluation-recognition: Evaluates the preformance of recognition model on the custom PV module dataset comprising of complex tables."
	@echo "  evaluation-extraction: Evaluates the tabular data extraction preformance of the pipeline on the custom PV module dataset comprising of complex tables."
	@echo "  evaluation-final: Evaluates the final relevant key data extraction preformance of the pipeline on the custom PV module dataset comprising of complex tables."
	@echo "  evaluate: Evaluates the entire pipeline."


finetune-detection:
	python src/Microsoft/main_version0_12thaugust.py --data_type detection --config_file src/detection_config.json --data_root_dir ../path/to/dataset --model_load_path weights/pubtables1m_detection_detr_r18.pth --load_weights_only --device cpu --model_save_dir weights/new

finetune-recognition:
	python src/Microsoft/main_version0_12thaugust.py --data_type structure --config_file src/latest_structure_config.json --data_root_dir ../path/to/dataset --model_load_path weights/pubtables1m_structure_detr_r18.pth --device cpu --load_weights_only --model_save_dir weights/new
	
inference-detection: clean
	python src/inference_4thDecemberfinal.py --mode detect --detection_config_path src/detection_config.json --detection_model_path weights/detection20_finetume_with_words.pth --detection_device cpu --in_dir inferences/pdf  --out_dir inferences/detectionOutput --crop_padding 25 -p

inference-recognition: inference-detection
	python src/inference_4thDecemberfinal.py --mode recognize --structure_config_path src/latest_structure_config_old.json --model_dir inferences/models --structure_model_path weights/exponentialLR_recognition_with_cannonical_finetuned_v2_model_20.pth --structure_device cpu --in_dir inferences/detectionOutput  --out_dir inferences/recognitionOutput -c -z --crop_padding 20

inference-extraction: clean
	python src/inference_4thDecemberfinal.py --mode extract --detection_config_path src/detection_config.json --detection_model_path weights/detection_finetuned_HPO_negatives.pth --detection_device cpu --structure_config_path src/latest_structure_config_old.json --structure_model_path weights/exponentialLR_recognition_with_cannonical_finetuned_v2_model_20.pth --structure_device cpu --in_dir inferences/pdf --out_dir inferences/extractionOutput --model_dir inferences/models -c -z --crop_padding 20

evaluation-detection:
	python src/Microsoft/TATR_detection_evaluation.py --data_type detection --config_file src/detection_config.json --data_root_dir Evaluation/solarModuleDetection --model_load_path weights/detection_finetuned_HPO_negatives.pth --device cpu

evaluation-recognition:
	python src/Microsoft/TATR_recognition_evaluation.py --mode eval --data_type structure --config_file src/latest_structure_config_old.json --data_root_dir Evaluation/solarModuleRecognition --model_load_path weights/with_mod_val_loss_model_structure50.pth --device cpu --table_words_dir Evaluation/solarModuleRecognition/words

evaluation-extraction:inference-extraction
	python src/evaluation_with_similarity_metrics_old.py 

evaluation-final:inference-extraction
	python src/evaluation_with_similarity_metrics_for_final_output.py 

evaluate: inference-extraction evaluation-detection evaluation-recognition evaluation-extraction evaluation-final

clean:
	rm -rf /app/inferences/detectionwords/*
	rm -rf /app/inferences/images/*
	rm -rf /app/inferences/pdf/extracted_tables.xlsx
