"""
Fine-tuning module for Whisper Large 3 with medical audio data.
"""
import os
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from ..config import ModelConfig, TrainingConfig, DataConfig
from ..medical_vocabulary import get_all_medical_terms


class WhisperDataCollator:
    """Data collator for Whisper fine-tuning."""
    
    def __init__(self, processor, model_config: ModelConfig):
        self.processor = processor
        self.model_config = model_config
        
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class WhisperFineTuner:
    """Fine-tuning handler for Whisper models."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        self.processor = None
        self.model = None
        self.metric = evaluate.load("wer")
        
    def load_model_and_processor(self):
        """Load Whisper model and processor."""
        print(f"Loading model: {self.model_config.model_name}")
        
        self.processor = WhisperProcessor.from_pretrained(
            self.model_config.model_name,
            language=self.model_config.language,
            task=self.model_config.task,
        )
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_config.model_name,
            torch_dtype=torch.float16 if self.model_config.compute_dtype == "float16" else torch.float32,
        )
        
        # Enable gradient checkpointing
        self.model.config.use_cache = False
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Force language and task tokens
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.model_config.language,
            task=self.model_config.task,
        )
        
        print(f"Model loaded on device: {self.model_config.device}")
        
    def prepare_dataset(self, dataset):
        """Prepare dataset for training."""
        
        def prepare_dataset_example(batch):
            # Load and resample audio
            audio = batch[self.data_config.audio_column]
            
            # Compute input features
            batch["input_features"] = self.processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"]
            ).input_features[0]
            
            # Encode target text
            batch["labels"] = self.processor.tokenizer(
                batch[self.data_config.text_column]
            ).input_ids
            
            return batch
        
        # Resample audio to 16kHz if needed
        dataset = dataset.cast_column(
            self.data_config.audio_column,
            Audio(sampling_rate=self.data_config.sample_rate)
        )
        
        # Apply preprocessing
        dataset = dataset.map(
            prepare_dataset_example,
            remove_columns=dataset.column_names,
            num_proc=self.training_config.dataloader_num_workers,
        )
        
        return dataset
    
    def compute_metrics(self, pred):
        """Compute WER metric."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def train(self):
        """Execute fine-tuning."""
        print("Starting fine-tuning process...")
        
        # Load model and processor
        self.load_model_and_processor()
        
        # Load datasets
        print("Loading datasets...")
        train_dataset = load_dataset(
            "audiofolder",
            data_dir=self.data_config.train_dataset_path,
            split="train"
        )
        eval_dataset = load_dataset(
            "audiofolder",
            data_dir=self.data_config.eval_dataset_path,
            split="train"
        )
        
        # Prepare datasets
        print("Preparing datasets...")
        train_dataset = self.prepare_dataset(train_dataset)
        eval_dataset = self.prepare_dataset(eval_dataset)
        
        # Data collator
        data_collator = WhisperDataCollator(self.processor, self.model_config)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            eval_steps=self.training_config.eval_steps,
            save_total_limit=self.training_config.save_total_limit,
            fp16=self.training_config.fp16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            push_to_hub=self.training_config.push_to_hub,
            report_to=self.training_config.report_to,
            evaluation_strategy="steps",
            predict_with_generate=True,
            generation_max_length=225,
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Train
        print("Training started...")
        trainer.train()
        
        # Save final model
        print(f"Saving model to {self.training_config.output_dir}")
        trainer.save_model(self.training_config.output_dir)
        self.processor.save_pretrained(self.training_config.output_dir)
        
        print("Training completed!")
        
        return trainer


def main():
    """Main entry point for fine-tuning."""
    from ..config import get_config_from_env
    
    model_config, training_config, data_config, _ = get_config_from_env()
    
    fine_tuner = WhisperFineTuner(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )
    
    fine_tuner.train()


if __name__ == "__main__":
    main()
