import numpy as np 
import pandas as pd

def generate_rule_feature_vector(rule, total_rules):
    """
    Generate a feature vector for a rule number.
    
    Args:
        rule (int): Rule number (0 to total_rules-1)
        total_rules (int): Total number of rules
        
    Returns:
        list: One-hot encoded vector for the rule
    """
    if rule < 0 or rule >= total_rules:
        raise ValueError(f"Rule number {rule} is out of the valid range 0-{total_rules-1}.")
    
    # Create a zero vector of length total_rules
    feature_vector = [0] * total_rules
    
    # Set the corresponding position to 1
    feature_vector[rule] = 1
    
    return feature_vector


