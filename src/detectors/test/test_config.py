# test_config.py
import yaml

with open('configs/detr.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
print("Config contents:")
for key, value in config.items():
    print(f"  {key}: {value} (type: {type(value).__name__})")