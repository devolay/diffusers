from sagemaker.huggingface import HuggingFace


hyperparameters={
    'pretrained_model_name_or_path': "runwayml/stable-diffusion-v1-5",
    "instance_prompt": "\"photo of sks man\"",
    "class_prompt": "\"photo of a man\"",
    "train_batch_size": 1,
    "max_train_steps": 800,
    "use_8bit_adam": True,
    "mixed_precision": "fp16",
    "hf_auth_token": "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",
    "lr_warmup_steps": 0,
}

huggingface_estimator = HuggingFace(
        entry_point='train_dreambooth_sagemaker.py',
        source_dir='.',
        instance_type='ml.g5.xlarge',
        instance_count=1,
        role='accelerate_sagemaker_execution_role',
        transformers_version='4.17',
        pytorch_version='1.10',
        py_version='py38',
        hyperparameters = hyperparameters
)

sagemaker_inputs = {
    "train": 's3://sagemaker-eu-west-1-485824217930/user_data'
}

huggingface_estimator.fit(sagemaker_inputs)

